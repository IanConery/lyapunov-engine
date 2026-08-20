import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from lyapunov_engine.ops import rmsnorm, swiglu, flash_attn_v2, paged_attention


@dataclass
class LlamaConfig:
    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE rotation to query or key tensor."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
        is_prefill: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        if is_prefill or k_cache is None:
            # Prefill phase: FlashAttention-2
            attn_out = flash_attn_v2(q, k, v, is_causal=True) # [batch, num_heads, seq_len, head_dim]
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.q_size)
        else:
            # Decode phase: PagedAttention
            q_dec = q.squeeze(2) # [batch, num_heads, head_dim]
            attn_out = paged_attention(q_dec, k_cache, v_cache, block_tables, context_lens)
            attn_out = attn_out.view(batch_size, 1, self.q_size)

        return self.o_proj(attn_out)


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        activated = swiglu(gate_up)
        return self.down_proj(activated)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.post_attention_layernorm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.rms_norm_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
        is_prefill: bool = True
    ) -> torch.Tensor:
        normed_input, _ = rmsnorm(hidden_states, self.input_layernorm_weight, self.eps)
        attn_out = self.self_attn(
            normed_input, cos, sin,
            k_cache=k_cache, v_cache=v_cache,
            block_tables=block_tables, context_lens=context_lens,
            is_prefill=is_prefill
        )
        hidden_states = hidden_states + attn_out

        normed_post, _ = rmsnorm(hidden_states, self.post_attention_layernorm_weight, self.eps)
        mlp_out = self.mlp(normed_post)
        hidden_states = hidden_states + mlp_out
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        dim = self.config.head_dim
        theta = self.config.rope_theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, dim]
        sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)
        return cos, sin

    def forward(
        self,
        input_ids: torch.Tensor,
        k_caches: Optional[list] = None,
        v_caches: Optional[list] = None,
        block_tables: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
        is_prefill: bool = True
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        seq_len = hidden_states.shape[1]
        cos, sin = self.get_cos_sin(seq_len, hidden_states.device, hidden_states.dtype)

        for i, layer in enumerate(self.layers):
            k_c = k_caches[i] if k_caches is not None else None
            v_c = v_caches[i] if v_caches is not None else None
            hidden_states = layer(
                hidden_states, cos, sin,
                k_cache=k_c, v_cache=v_c,
                block_tables=block_tables, context_lens=context_lens,
                is_prefill=is_prefill
            )

        normed, _ = rmsnorm(hidden_states, self.norm_weight, self.config.rms_norm_eps)
        logits = self.lm_head(normed)
        return logits
