import pytest
import torch
import torch.nn as nn
from lyapunov_engine.engine.cuda_graph_runner import CUDAGraphRunner
from lyapunov_engine.models.llama import LlamaConfig, LlamaForCausalLM


def test_cuda_graph_runner_bucketing():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64
    )
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.float32)

    runner = CUDAGraphRunner(
        model=model,
        batch_buckets=[1, 2, 4, 8],
        device=str(device),
        dtype=torch.float32
    )

    # Capture graphs up to bucket 4
    runner.capture_graphs(max_bucket=4)

    # Test batch size 1 (exact bucket)
    tokens_1 = torch.randint(0, 100, (1, 1), device=device)
    out_1 = runner.forward(tokens_1)
    assert out_1.shape == (1, 1, 128)

    # Test batch size 3 (pads to bucket 4 and slices back to 3)
    tokens_3 = torch.randint(0, 100, (3, 1), device=device)
    out_3 = runner.forward(tokens_3)
    assert out_3.shape == (3, 1, 128)
