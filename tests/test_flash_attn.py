import pytest
import torch
from lyapunov_engine.ops import flash_attn_v2, flash_decoding


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (16, 4)])  # MHA & GQA
@pytest.mark.parametrize("q_len,kv_len", [(64, 64), (128, 128), (64, 256)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_flash_attn_v2_numerical_parity(
    batch_size, num_heads, num_kv_heads, q_len, kv_len, head_dim, dtype
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q = torch.randn(
        (batch_size, num_heads, q_len, head_dim), dtype=dtype, device=device
    )
    k = torch.randn(
        (batch_size, num_kv_heads, kv_len, head_dim), dtype=dtype, device=device
    )
    v = torch.randn(
        (batch_size, num_kv_heads, kv_len, head_dim), dtype=dtype, device=device
    )

    # Reference SDPA with causal masking
    scale = 1.0 / (head_dim**0.5)
    gqa_ratio = num_heads // num_kv_heads
    k_ref = k.repeat_interleave(gqa_ratio, dim=1).float()
    v_ref = v.repeat_interleave(gqa_ratio, dim=1).float()
    q_ref = q.float()

    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=(q_len == kv_len), scale=scale
    ).to(dtype)

    # Custom FlashAttention-2
    out = flash_attn_v2(q, k, v, sm_scale=scale, is_causal=(q_len == kv_len))

    atol = 2e-3 if dtype == torch.float16 else 1e-4
    rtol = 2e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (16, 4)])
@pytest.mark.parametrize("kv_len", [128, 512, 1024])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_flash_decoding_parity(
    batch_size, num_heads, num_kv_heads, kv_len, head_dim, dtype
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q = torch.randn((batch_size, num_heads, 1, head_dim), dtype=dtype, device=device)
    k = torch.randn(
        (batch_size, num_kv_heads, kv_len, head_dim), dtype=dtype, device=device
    )
    v = torch.randn(
        (batch_size, num_kv_heads, kv_len, head_dim), dtype=dtype, device=device
    )

    scale = 1.0 / (head_dim**0.5)
    gqa_ratio = num_heads // num_kv_heads
    k_ref = k.repeat_interleave(gqa_ratio, dim=1).float()
    v_ref = v.repeat_interleave(gqa_ratio, dim=1).float()
    q_ref = q.float()

    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=False, scale=scale
    ).to(dtype)

    out = flash_decoding(q, k, v, num_partitions=4, sm_scale=scale)

    atol = 2e-3 if dtype == torch.float16 else 1e-4
    rtol = 2e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)
