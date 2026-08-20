from typing import Optional, Tuple
import torch

try:
    from lyapunov_engine import _C
    HAS_CUDA_EXT = True
except ImportError:
    _C = None
    HAS_CUDA_EXT = False


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    residual: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fused RMSNorm with optional residual addition.
    
    Args:
        x: Input tensor of shape [..., hidden_dim]
        weight: Gamma scaling weights of shape [hidden_dim]
        eps: Small constant for numerical stability
        residual: Optional residual tensor of shape [..., hidden_dim]
        
    Returns:
        (out, out_residual): Normalized output and accumulated residual
    """
    if HAS_CUDA_EXT and x.is_cuda and weight.is_cuda:
        orig_shape = x.shape
        x_2d = x.contiguous().view(-1, orig_shape[-1])
        res_2d = residual.contiguous().view(-1, orig_shape[-1]) if residual is not None else None
        res = _C.rmsnorm_forward(x_2d, weight.contiguous(), eps, res_2d)
        out = res[0].view(orig_shape)
        out_res = res[1].view(orig_shape) if len(res) > 1 else None
        return out, out_res
    
    # Eager reference implementation
    if residual is not None:
        x = x + residual
        out_res = x
    else:
        out_res = None
        
    variance = x.pow(2).mean(-1, keepdim=True)
    out = x * torch.rsqrt(variance + eps) * weight
    return out, out_res


def swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU activation function: out = SiLU(gate) * up.
    
    Args:
        gate_up: Input tensor where last dim has size 2 * intermediate_dim
        
    Returns:
        Tensor of shape [..., intermediate_dim]
    """
    if HAS_CUDA_EXT and gate_up.is_cuda:
        orig_shape = gate_up.shape
        last_dim = orig_shape[-1]
        gate_up_2d = gate_up.contiguous().view(-1, last_dim)
        out_2d = _C.swiglu_forward(gate_up_2d)
        new_shape = list(orig_shape[:-1]) + [last_dim // 2]
        return out_2d.view(new_shape)
    
    # Eager reference implementation
    gate, up = gate_up.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def flash_attn_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: Optional[float] = None,
    is_causal: bool = True
) -> torch.Tensor:
    """FlashAttention-2 forward pass for sequence prefill.
    
    Args:
        q: Query tensor [batch, num_heads, seqlen, head_dim]
        k: Key tensor [batch, num_kv_heads, seqlen, head_dim]
        v: Value tensor [batch, num_kv_heads, seqlen, head_dim]
        sm_scale: Softmax scale factor (default: 1 / sqrt(head_dim))
        is_causal: Whether to apply causal masking
        
    Returns:
        Output tensor [batch, num_heads, seqlen, head_dim]
    """
    if HAS_CUDA_EXT and q.is_cuda:
        return _C.flash_attn_v2_fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            sm_scale,
            is_causal
        )
    
    # Eager reference implementation
    head_dim = q.shape[-1]
    scale = sm_scale if sm_scale is not None else 1.0 / (head_dim ** 0.5)
    
    # Expand GQA if necessary
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_heads != num_kv_heads:
        ratio = num_heads // num_kv_heads
        k = k.repeat_interleave(ratio, dim=1)
        v = v.repeat_interleave(ratio, dim=1)
        
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        q_len, k_len = q.shape[2], k.shape[2]
        mask = torch.triu(torch.ones(q_len, k_len, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -1e9)
        
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def flash_decoding(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_partitions: int = 4,
    sm_scale: Optional[float] = None
) -> torch.Tensor:
    """FlashDecoding split-KV kernel for single-token generation.
    
    Args:
        q: Query tensor [batch, num_heads, 1, head_dim]
        k: Key tensor [batch, num_kv_heads, kv_seqlen, head_dim]
        v: Value tensor [batch, num_kv_heads, kv_seqlen, head_dim]
        num_partitions: Number of KV partitions across sequence length
        sm_scale: Softmax scale factor
        
    Returns:
        Output tensor [batch, num_heads, 1, head_dim]
    """
    if HAS_CUDA_EXT and q.is_cuda:
        return _C.flash_decoding(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            num_partitions,
            sm_scale
        )
    return flash_attn_v2(q, k, v, sm_scale=sm_scale, is_causal=False)


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    sm_scale: Optional[float] = None
) -> torch.Tensor:
    """PagedAttention decoding operator with block table indirection.
    
    Args:
        q: Query tensor [batch, num_heads, head_dim]
        k_cache: Key cache pool [num_blocks, num_kv_heads, block_size, head_dim]
        v_cache: Value cache pool [num_blocks, num_kv_heads, block_size, head_dim]
        block_tables: Block index table [batch, max_blocks_per_seq] (int32)
        context_lens: Context lengths per request [batch] (int32)
        sm_scale: Softmax scale factor
        
    Returns:
        Output tensor [batch, num_heads, head_dim]
    """
    if HAS_CUDA_EXT and q.is_cuda:
        return _C.paged_attention_v1(
            q.contiguous(),
            k_cache.contiguous(),
            v_cache.contiguous(),
            block_tables.to(torch.int32).contiguous(),
            context_lens.to(torch.int32).contiguous(),
            sm_scale
        )
    
    # Eager reference fallback
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    block_size = k_cache.shape[2]
    scale = sm_scale if sm_scale is not None else 1.0 / (head_dim ** 0.5)
    
    out = torch.zeros_like(q)
    gqa_ratio = num_heads // num_kv_heads
    
    for b in range(batch_size):
        c_len = int(context_lens[b].item())
        if c_len == 0:
            continue
        num_blocks = (c_len + block_size - 1) // block_size
        b_table = block_tables[b, :num_blocks].tolist()
        
        # Assemble contiguous K and V
        k_slices = [k_cache[b_idx] for b_idx in b_table]
        v_slices = [v_cache[b_idx] for b_idx in b_table]
        
        k_seq = torch.cat(k_slices, dim=1)[:, :c_len, :]  # [num_kv_heads, c_len, head_dim]
        v_seq = torch.cat(v_slices, dim=1)[:, :c_len, :]
        
        for h in range(num_heads):
            kv_h = h // gqa_ratio
            q_h = q[b, h] # [head_dim]
            k_h = k_seq[kv_h] # [c_len, head_dim]
            v_h = v_seq[kv_h] # [c_len, head_dim]
            
            scores = torch.matmul(k_h, q_h) * scale # [c_len]
            weights = torch.softmax(scores, dim=0) # [c_len]
            out[b, h] = torch.matmul(weights, v_h)
            
    return out
