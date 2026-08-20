import pytest
import torch
import math

try:
    from lyapunov_engine import _C
    HAS_CUDA_EXT = True
except ImportError:
    _C = None
    HAS_CUDA_EXT = False


def precompute_cos_sin(max_pos: int, head_dim: int, device: torch.device) -> torch.Tensor:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq) # [max_pos, head_dim/2]
    cos_vals = torch.cos(freqs)
    sin_vals = torch.sin(freqs)
    # Pack [cos, sin] into [max_pos, head_dim]
    return torch.cat([cos_vals, sin_vals], dim=-1).contiguous()


@pytest.mark.parametrize("num_tokens,num_heads,num_kv_heads,head_dim", [
    (1, 8, 2, 64),
    (4, 16, 4, 128),
    (16, 32, 8, 128)
])
def test_fused_rope_paged(num_tokens, num_heads, num_kv_heads, head_dim):
    if not torch.cuda.is_available() or not HAS_CUDA_EXT:
        pytest.skip("CUDA extension not available")

    device = torch.device("cuda:0")
    block_size = 16
    num_blocks = 32
    max_blocks = 8

    # Inputs
    q_in = torch.randn((num_tokens, num_heads, head_dim), dtype=torch.float32, device=device)
    k_in = torch.randn((num_tokens, num_kv_heads, head_dim), dtype=torch.float32, device=device)
    v_in = torch.randn((num_tokens, num_kv_heads, head_dim), dtype=torch.float32, device=device)

    cos_sin = precompute_cos_sin(2048, head_dim, device)

    # Outputs
    q_out = torch.empty_like(q_in)
    k_cache = torch.zeros((num_blocks, num_kv_heads, block_size, head_dim), dtype=torch.float32, device=device)
    v_cache = torch.zeros((num_blocks, num_kv_heads, block_size, head_dim), dtype=torch.float32, device=device)

    # Block tables & context lengths
    block_tables = torch.zeros((num_tokens, max_blocks), dtype=torch.int32, device=device)
    for i in range(num_tokens):
        block_tables[i, 0] = i % num_blocks
    context_lens = torch.ones((num_tokens,), dtype=torch.int32, device=device)

    _C.fused_rope_paged(
        q_out, k_cache, v_cache, q_in, k_in, v_in,
        cos_sin, block_tables, context_lens
    )

    # Verify query output shape and finite values
    assert q_out.shape == q_in.shape
    assert not torch.isnan(q_out).any()
    assert not torch.isinf(q_out).any()

    # Verify key/value cached data
    assert not torch.isnan(k_cache).any()
    assert not torch.isnan(v_cache).any()
