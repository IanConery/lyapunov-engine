import argparse
import torch
import numpy as np
from lyapunov_engine.models.llama import LlamaConfig, LlamaForCausalLM
from lyapunov_engine.engine.lyapunov import compute_lyapunov_divergence


def run_lyapunov_benchmark(context_lens=[32, 128, 512, 1024], num_steps=16):
    print("=" * 70)
    print("Benchmarking Lyapunov Exponent & Trajectory Divergence")
    print("=" * 70)
    print(f"{'Context Len':<15} | {'Lyapunov Exp (λ)':<20} | {'Max Divergence D(t)':<20} | {'Status':<15}")
    print("-" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = LlamaConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32
    )

    model = LlamaForCausalLM(config).to(device=device, dtype=torch.float32)

    for seq_len in context_lens:
        tokens = torch.randint(0, 1000, (seq_len,)).tolist()
        lambda_val, div_history = compute_lyapunov_divergence(
            model=model,
            prompt_tokens=tokens,
            num_steps=num_steps,
            device=device
        )
        max_div = max(div_history) if div_history else 0.0
        status = "Chaotic" if lambda_val > 0.0 else "Stable"
        print(f"{seq_len:<15} | {lambda_val:<20.4f} | {max_div:<20.4f} | {status:<15}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Lyapunov Exponent Profiler")
    parser.add_argument("--context-lens", type=int, nargs="+", default=[32, 128, 512, 1024])
    parser.add_argument("--num-steps", type=int, default=16)
    args = parser.parse_args()
    run_lyapunov_benchmark(context_lens=args.context_lens, num_steps=args.num_steps)


if __name__ == "__main__":
    main()
