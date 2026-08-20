import argparse

import torch
from lyapunov_engine.ops import flash_attn_v2, rmsnorm


def benchmark_op(fn, warmup: int = 25, iters: int = 100) -> float:
    """Benchmark a function on CUDA using torch.cuda.Event and return median latency in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [
        s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)
    ]  # microseconds
    times.sort()
    return times[len(times) // 2]


def run_rmsnorm_benchmarks():
    print("=" * 70)
    print("Benchmarking Fused RMSNorm vs PyTorch Eager (FP16)")
    print("=" * 70)
    print(
        f"{'Hidden Dim':<12} | {'Tokens':<10} | {'Eager (us)':<12} | {'Fused (us)':<12} | {'Speedup':<8}"
    )
    print("-" * 70)

    device = torch.device("cuda:0")
    for hidden_dim in [2048, 4096, 8192]:
        for tokens in [32, 256, 1024, 4096]:
            x = torch.randn((tokens, hidden_dim), dtype=torch.float16, device=device)
            weight = torch.randn(hidden_dim, dtype=torch.float16, device=device)
            eps = 1e-5

            def eager_fn(curr_x=x, curr_w=weight, curr_eps=eps):
                var = curr_x.float().pow(2).mean(-1, keepdim=True)
                return (
                    curr_x.float() * torch.rsqrt(var + curr_eps) * curr_w.float()
                ).half()

            def fused_fn(curr_x=x, curr_w=weight, curr_eps=eps):
                return rmsnorm(curr_x, curr_w, curr_eps)

            eager_us = benchmark_op(eager_fn)
            fused_us = benchmark_op(fused_fn)
            speedup = eager_us / fused_us

            print(
                f"{hidden_dim:<12} | {tokens:<10} | {eager_us:<12.2f} | {fused_us:<12.2f} | {speedup:<8.2f}x"
            )


def run_flash_attn_benchmarks():
    print("\n" + "=" * 70)
    print("Benchmarking FlashAttention-2 vs PyTorch SDPA (FP16, Head Dim 128)")
    print("=" * 70)
    print(
        f"{'Seq Len':<10} | {'Heads':<8} | {'SDPA (us)':<12} | {'FlashAttn (us)':<15} | {'Speedup':<8}"
    )
    print("-" * 70)

    device = torch.device("cuda:0")
    head_dim = 128
    batch_size = 1
    num_heads = 32
    num_kv_heads = 8

    for seq_len in [512, 1024, 2048, 4096]:
        q = torch.randn(
            (batch_size, num_heads, seq_len, head_dim),
            dtype=torch.float16,
            device=device,
        )
        k = torch.randn(
            (batch_size, num_kv_heads, seq_len, head_dim),
            dtype=torch.float16,
            device=device,
        )
        v = torch.randn(
            (batch_size, num_kv_heads, seq_len, head_dim),
            dtype=torch.float16,
            device=device,
        )

        k_rep = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_rep = v.repeat_interleave(num_heads // num_kv_heads, dim=1)

        def sdpa_fn(curr_q=q, curr_k=k_rep, curr_v=v_rep):
            return torch.nn.functional.scaled_dot_product_attention(
                curr_q, curr_k, curr_v, is_causal=True
            )

        def flash_fn(curr_q=q, curr_k=k, curr_v=v):
            return flash_attn_v2(curr_q, curr_k, curr_v, is_causal=True)

        sdpa_us = benchmark_op(sdpa_fn)
        flash_us = benchmark_op(flash_fn)
        speedup = sdpa_us / flash_us

        print(
            f"{seq_len:<10} | {num_heads:<8} | {sdpa_us:<12.2f} | {flash_us:<15.2f} | {speedup:<8.2f}x"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
    else:
        run_rmsnorm_benchmarks()
        run_flash_attn_benchmarks()
