import pytest
import torch
from lyapunov_engine.ops import rmsnorm, swiglu


@pytest.mark.parametrize("hidden_dim", [256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("num_tokens", [1, 16, 128, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rmsnorm_numerical_parity(hidden_dim, num_tokens, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    x = torch.randn((num_tokens, hidden_dim), dtype=dtype, device=device)
    weight = torch.randn(hidden_dim, dtype=dtype, device=device)
    eps = 1e-5

    # PyTorch reference
    variance = x.float().pow(2).mean(-1, keepdim=True)
    ref_out = (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(dtype)

    # Custom op
    out, _ = rmsnorm(x, weight, eps)

    atol = 1e-3 if dtype == torch.float16 else 1e-5
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("hidden_dim", [512, 2048])
@pytest.mark.parametrize("num_tokens", [32, 256])
def test_rmsnorm_with_residual(hidden_dim, num_tokens):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    x = torch.randn((num_tokens, hidden_dim), dtype=torch.float32, device=device)
    res = torch.randn((num_tokens, hidden_dim), dtype=torch.float32, device=device)
    weight = torch.randn(hidden_dim, dtype=torch.float32, device=device)
    eps = 1e-5

    # Reference
    accum_res = x + res
    variance = accum_res.pow(2).mean(-1, keepdim=True)
    ref_out = accum_res * torch.rsqrt(variance + eps) * weight

    # Custom op
    out, out_res = rmsnorm(x, weight, eps, residual=res)

    torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_res, accum_res, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("intermediate_dim", [512, 2048, 8192])
@pytest.mark.parametrize("num_tokens", [1, 32, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_swiglu_numerical_parity(intermediate_dim, num_tokens, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    gate_up = torch.randn(
        (num_tokens, 2 * intermediate_dim), dtype=dtype, device=device
    )

    # Reference
    gate, up = gate_up.chunk(2, dim=-1)
    ref_out = (torch.nn.functional.silu(gate.float()) * up.float()).to(dtype)

    # Custom op
    out = swiglu(gate_up)

    atol = 1e-3 if dtype == torch.float16 else 1e-5
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)
