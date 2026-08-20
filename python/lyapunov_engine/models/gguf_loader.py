import os

import gguf
import numpy as np
import torch

from lyapunov_engine.models.llama import LlamaConfig, LlamaForCausalLM


def load_llama_config_from_gguf(reader: gguf.GGUFReader) -> LlamaConfig:
    """Extract model hyperparameter metadata from GGUFReader into LlamaConfig."""
    fields = {f.name: f for f in reader.fields.values()}

    def get_field(name: str, default=None):
        if name in fields:
            val = fields[name].parts[-1]
            if isinstance(val, np.ndarray) and val.size == 1:
                return val.item()
            return val
        return default

    vocab_size = get_field("llama.vocab_size", get_field("general.vocab_size", 128256))
    hidden_size = get_field("llama.embedding_length", 2048)
    intermediate_size = get_field("llama.feed_forward_length", 8192)
    num_hidden_layers = get_field("llama.block_count", 16)
    num_attention_heads = get_field("llama.attention.head_count", 32)
    num_key_value_heads = get_field("llama.attention.head_count_kv", 8)
    head_dim = hidden_size // num_attention_heads if num_attention_heads else 64
    max_position_embeddings = get_field("llama.context_length", 8192)
    rms_norm_eps = get_field("llama.attention.layer_norm_rms_epsilon", 1e-5)
    rope_theta = get_field("llama.rope.freq_base", 500000.0)

    return LlamaConfig(
        vocab_size=int(vocab_size),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_hidden_layers=int(num_hidden_layers),
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
        head_dim=int(head_dim),
        max_position_embeddings=int(max_position_embeddings),
        rms_norm_eps=float(rms_norm_eps),
        rope_theta=float(rope_theta),
    )


def load_gguf_model(
    gguf_path: str, device: str = "cpu", dtype: torch.dtype = torch.float16
) -> LlamaForCausalLM:
    """Load quantized model from GGUF binary file directly into LlamaForCausalLM."""
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF file not found at: {gguf_path}")

    reader = gguf.GGUFReader(gguf_path)
    config = load_llama_config_from_gguf(reader)
    model = LlamaForCausalLM(config).to(device=device, dtype=dtype)

    # Tensor mapping dictionary
    tensor_map: dict[str, gguf.ReaderTensor] = {t.name: t for t in reader.tensors}

    # 1. Load Token Embeddings & Output Norm
    if "token_embd.weight" in tensor_map:
        t = tensor_map["token_embd.weight"]
        emb_data = torch.from_numpy(t.data.copy()).to(dtype=dtype, device=device)
        model.embed_tokens.weight.data.copy_(emb_data)

    if "output_norm.weight" in tensor_map:
        t = tensor_map["output_norm.weight"]
        norm_data = torch.from_numpy(t.data.copy()).to(dtype=dtype, device=device)
        model.norm_weight.data.copy_(norm_data)

    if "output.weight" in tensor_map:
        t = tensor_map["output.weight"]
        out_data = torch.from_numpy(t.data.copy()).to(dtype=dtype, device=device)
        model.lm_head.weight.data.copy_(out_data)
    elif "token_embd.weight" in tensor_map:
        model.lm_head.weight.data.copy_(model.embed_tokens.weight.data)

    # 2. Load Layer Tensors
    for i in range(config.num_hidden_layers):
        layer = model.layers[i]
        prefix = f"blk.{i}."

        # Input Layernorm & Post Attention Layernorm
        attn_norm_name = f"{prefix}attn_norm.weight"
        if attn_norm_name in tensor_map:
            t = tensor_map[attn_norm_name]
            layer.input_layernorm_weight.data.copy_(
                torch.from_numpy(t.data.copy()).to(dtype=dtype, device=device)
            )

        ffn_norm_name = f"{prefix}ffn_norm.weight"
        if ffn_norm_name in tensor_map:
            t = tensor_map[ffn_norm_name]
            layer.post_attention_layernorm_weight.data.copy_(
                torch.from_numpy(t.data.copy()).to(dtype=dtype, device=device)
            )

        # Linear projections
        for proj_name, layer_proj in [
            (f"{prefix}attn_q.weight", layer.self_attn.q_proj),
            (f"{prefix}attn_k.weight", layer.self_attn.k_proj),
            (f"{prefix}attn_v.weight", layer.self_attn.v_proj),
            (f"{prefix}attn_output.weight", layer.self_attn.o_proj),
            (f"{prefix}ffn_down.weight", layer.mlp.down_proj),
        ]:
            if proj_name in tensor_map:
                t = tensor_map[proj_name]
                w_data = torch.from_numpy(t.data.copy()).to(dtype=dtype, device=device)
                layer_proj.weight.data.copy_(w_data)

        # Fused gate_up projection
        gate_name = f"{prefix}ffn_gate.weight"
        up_name = f"{prefix}ffn_up.weight"
        if gate_name in tensor_map and up_name in tensor_map:
            t_gate = tensor_map[gate_name]
            t_up = tensor_map[up_name]
            gate_w = torch.from_numpy(t_gate.data.copy()).to(dtype=dtype, device=device)
            up_w = torch.from_numpy(t_up.data.copy()).to(dtype=dtype, device=device)
            fused = torch.cat([gate_w, up_w], dim=0)
            layer.mlp.gate_up_proj.weight.data.copy_(fused)

    model.eval()
    return model
