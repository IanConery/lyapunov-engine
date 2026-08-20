import argparse
import sys
import torch
from transformers import AutoTokenizer

from lyapunov_engine.models.weight_loader import from_pretrained_llama
from lyapunov_engine.engine.llm_engine import LLMEngine, EngineConfig
from lyapunov_engine.engine.sampling import SamplingParams
from lyapunov_engine.server.api_server import create_app


def run_generate(args):
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading weights: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = from_pretrained_llama(args.model, dtype=dtype, device=device)

    config = EngineConfig(
        model_config=model.config,
        block_size=args.block_size,
        num_gpu_blocks=args.num_blocks,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_batched_tokens,
        dtype=dtype,
        device=device
    )

    engine = LLMEngine(model, config)
    prompt_tokens = tokenizer.encode(args.prompt)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    engine.add_request("req-0", prompt_tokens, sampling)
    print("\n--- Output Generation ---")

    while engine.has_unfinished_requests():
        outputs = engine.step()
        for out in outputs:
            if out.is_finished:
                text = tokenizer.decode(out.output_tokens, skip_special_tokens=True)
                print(f"{text}")


def run_serve(args):
    import uvicorn

    print(f"Initializing Lyapunov Engine server for: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = from_pretrained_llama(args.model, dtype=dtype, device=device)
    config = EngineConfig(
        model_config=model.config,
        block_size=args.block_size,
        num_gpu_blocks=args.num_blocks,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_batched_tokens,
        dtype=dtype,
        device=device
    )

    engine = LLMEngine(model, config)
    app = create_app(engine=engine, tokenizer=tokenizer, model_name=args.model)

    print(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(
        description="Lyapunov Engine: Custom CUDA LLM Inference Engine with Dynamic Trajectory & Uncertainty Analysis"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Run text generation from prompt")
    gen_parser.add_argument("--model", type=str, required=True, help="HuggingFace model repo or local directory")
    gen_parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    gen_parser.add_argument("--max-tokens", type=int, default=64)
    gen_parser.add_argument("--temperature", type=float, default=0.7)
    gen_parser.add_argument("--top-p", type=float, default=1.0)
    gen_parser.add_argument("--block-size", type=int, default=16)
    gen_parser.add_argument("--num-blocks", type=int, default=512)
    gen_parser.add_argument("--max-num-seqs", type=int, default=16)
    gen_parser.add_argument("--max-batched-tokens", type=int, default=2048)

    # Serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible API server")
    serve_parser.add_argument("--model", type=str, required=True, help="HuggingFace model repo or local directory")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--block-size", type=int, default=16)
    serve_parser.add_argument("--num-blocks", type=int, default=1024)
    serve_parser.add_argument("--max-num-seqs", type=int, default=32)
    serve_parser.add_argument("--max-batched-tokens", type=int, default=4096)

    args = parser.parse_args()

    if args.subcommand == "generate":
        run_generate(args)
    elif args.subcommand == "serve":
        run_serve(args)


if __name__ == "__main__":
    main()
