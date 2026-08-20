import pytest
import torch
import torch.nn as nn
from lyapunov_engine.models.quant import QuantizedLinear, QuantType
from lyapunov_engine.models.fp8_loader import quantize_to_fp8_e4m3, convert_linear_to_fp8
from lyapunov_engine.models.marlin_loader import quantize_and_pack_marlin_w4a16, convert_linear_to_marlin_w4a16

try:
    from lyapunov_engine import _C
    HAS_CUDA_EXT = True
except ImportError:
    _C = None
    HAS_CUDA_EXT = False


def create_synthetic_q4_0_bytes(num_blocks: int, device: torch.device) -> torch.Tensor:
    """Create valid Q4_0 bytes with valid FP16 scale factors."""
    scales = (torch.rand(num_blocks, dtype=torch.float16, device=device) * 0.1 + 0.01).view(torch.uint8) # 2 bytes each
    scales = scales.view(num_blocks, 2)
    nibbles = torch.randint(0, 255, (num_blocks, 16), dtype=torch.uint8, device=device)
    blocks = torch.cat([scales, nibbles], dim=1) # [num_blocks, 18]
    return blocks.view(-1)


def create_synthetic_q8_0_bytes(num_blocks: int, device: torch.device) -> torch.Tensor:
    """Create valid Q8_0 bytes with valid FP16 scale factors."""
    scales = (torch.rand(num_blocks, dtype=torch.float16, device=device) * 0.1 + 0.01).view(torch.uint8)
    scales = scales.view(num_blocks, 2)
    int8s = torch.randint(0, 255, (num_blocks, 32), dtype=torch.uint8, device=device)
    blocks = torch.cat([scales, int8s], dim=1) # [num_blocks, 34]
    return blocks.view(-1)


@pytest.mark.parametrize("in_features,out_features", [(64, 128), (256, 512), (1024, 1024)])
def test_q4_0_dequantize_and_gemv(in_features, out_features):
    if not torch.cuda.is_available() or not HAS_CUDA_EXT:
        pytest.skip("CUDA extension not available")

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    num_blocks = (out_features * in_features) // 32
    raw_bytes = create_synthetic_q4_0_bytes(num_blocks, device)

    # Dequantize with custom kernel
    dequant = _C.dequantize_q4_0(raw_bytes, out_features * in_features).view(out_features, in_features)
    assert dequant.shape == (out_features, in_features)
    assert not torch.isnan(dequant).any()

    # Test Quantized GEMV
    x = torch.randn((2, in_features), dtype=torch.float32, device=device)
    out = _C.quant_gemv(x, raw_bytes, None, in_features, out_features, int(QuantType.Q4_0))
    ref_out = torch.matmul(x, dequant.t())

    assert torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("in_features,out_features", [(64, 128), (256, 512), (1024, 1024)])
def test_q8_0_dequantize_and_gemv(in_features, out_features):
    if not torch.cuda.is_available() or not HAS_CUDA_EXT:
        pytest.skip("CUDA extension not available")

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    num_blocks = (out_features * in_features) // 32
    raw_bytes = create_synthetic_q8_0_bytes(num_blocks, device)

    dequant = _C.dequantize_q8_0(raw_bytes, out_features * in_features).view(out_features, in_features)
    assert dequant.shape == (out_features, in_features)
    assert not torch.isnan(dequant).any()

    x = torch.randn((2, in_features), dtype=torch.float32, device=device)
    out = _C.quant_gemv(x, raw_bytes, None, in_features, out_features, int(QuantType.Q8_0))
    ref_out = torch.matmul(x, dequant.t())

    assert torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("in_features,out_features", [(64, 128), (256, 512), (1024, 1024)])
def test_marlin_w4a16_gemm(in_features, out_features):
    if not torch.cuda.is_available() or not HAS_CUDA_EXT:
        pytest.skip("CUDA extension not available")

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    linear = nn.Linear(in_features, out_features, bias=False, device=device)
    qlinear = convert_linear_to_marlin_w4a16(linear, device=device)

    x = torch.randn((4, in_features), dtype=torch.float32, device=device)
    out = qlinear(x)

    assert out.shape == (4, out_features)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize("in_features,out_features", [(64, 128), (256, 512), (1024, 1024)])
def test_fp8_gemm(in_features, out_features):
    if not torch.cuda.is_available() or not HAS_CUDA_EXT:
        pytest.skip("CUDA extension not available")

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    linear = nn.Linear(in_features, out_features, bias=False, device=device)
    qlinear = convert_linear_to_fp8(linear, device=device)

    x = torch.randn((4, in_features), dtype=torch.float32, device=device)
    out = qlinear(x)

    assert out.shape == (4, out_features)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
