import time
import argparse
import torch
from lyapunov_engine.models.llama import LlamaForCausalLM, LlamaConfig
from lyapunov_engine.engine.llm_engine import LLMEngine
from lyapunov_engine.engine.sampling import SamplingParams


def run_serving_benchmark(batch_sizes=[1, 4, 8, 16], prompt_len=128, gen_tokens=32):
    print("=" * 80)
    print(f"Benchmarking End-to-End Serving Engine (Prompt: {prompt_len} toks, Gen: {gen_tokens} toks)")
    print("=" * 80)
    print(f"{'Batch Size':<12} | {'TTFT (ms)':<12} | {'ITL (ms)':<12} | {'Throughput (tok/s)':<20}")
    print("-" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=64
    )

    model = LlamaForCausalLM(config).half().to(device)

    for b in batch_sizes:
        engine = LLMEngine(
            model=model,
            num_blocks=2048,
            block_size=16,
            max_num_seqs=b,
            max_num_batched_tokens=4096,
            device=device
        )

        sampling_params = SamplingParams(max_tokens=gen_tokens, temperature=0.0)

        # Submit batch requests
        req_ids = []
        for _ in range(b):
            prompt = [100 + i for i in range(prompt_len)]
            req_ids.append(engine.add_request(prompt, sampling_params))

        # Time TTFT (Prefill)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefill_out = engine.step()
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        # Time Decode steps
        decode_times = []
        total_gen = 0
        while engine.has_unfinished_requests():
            torch.cuda.synchronize()
            t_step = time.perf_counter()
            step_out = engine.step()
            torch.cuda.synchronize()
            decode_times.append((time.perf_counter() - t_step) * 1000.0)
            total_gen += len(step_out)

        avg_itl_ms = sum(decode_times) / len(decode_times) if decode_times else 0.0
        total_time_s = (ttft_ms + sum(decode_times)) / 1000.0
        total_tokens = (b * prompt_len) + total_gen
        throughput = total_tokens / total_time_s if total_time_s > 0 else 0.0

        print(f"{b:<12} | {ttft_ms:<12.2f} | {avg_itl_ms:<12.2f} | {throughput:<20.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    args = parser.parse_args()

    run_serving_benchmark(batch_sizes=args.batch_sizes)
