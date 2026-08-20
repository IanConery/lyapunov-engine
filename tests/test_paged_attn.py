import pytest
import torch
from lyapunov_engine.ops import paged_attention


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (16, 4)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_paged_attention_numerical_parity(batch_size, num_heads, num_kv_heads, head_dim, block_size, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    num_blocks = 128
    k_cache = torch.randn((num_blocks, num_kv_heads, block_size, head_dim), dtype=dtype, device=device)
    v_cache = torch.randn((num_blocks, num_kv_heads, block_size, head_dim), dtype=dtype, device=device)

    q = torch.randn((batch_size, num_heads, head_dim), dtype=dtype, device=device)

    context_lens = torch.tensor([15, 32, 48, 60][:batch_size], dtype=torch.int32, device=device)
    max_blocks = 4
    block_tables = torch.randint(0, num_blocks, (batch_size, max_blocks), dtype=torch.int32, device=device)

    scale = 1.0 / (head_dim ** 0.5)

    # Reference computation
    gqa_ratio = num_heads // num_kv_heads
    ref_out = torch.zeros_like(q)

    for b in range(batch_size):
        c_len = int(context_lens[b].item())
        n_blks = (c_len + block_size - 1) // block_size
        b_tbl = block_tables[b, :n_blks].tolist()

        k_slices = [k_cache[idx] for idx in b_tbl]
        v_slices = [v_cache[idx] for idx in b_tbl]

        k_seq = torch.cat(k_slices, dim=1)[:, :c_len, :].float()
        v_seq = torch.cat(v_slices, dim=1)[:, :c_len, :].float()

        for h in range(num_heads):
            kv_h = h // gqa_ratio
            q_h = q[b, h].float()
            k_h = k_seq[kv_h]
            v_h = v_seq[kv_h]

            scores = torch.matmul(k_h, q_h) * scale
            weights = torch.softmax(scores, dim=0)
            ref_out[b, h] = torch.matmul(weights, v_h).to(dtype)

    # Custom PagedAttention op
    out = paged_attention(q, k_cache, v_cache, block_tables, context_lens, sm_scale=scale)

    atol = 2e-3 if dtype == torch.float16 else 1e-4
    rtol = 2e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)
