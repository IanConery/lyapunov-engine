from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from lyapunov_engine.models.quant import QuantizedLinear, QuantType


def quantize_to_fp8_e4m3(
    weight: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    """Quantize an FP16/FP32 weight tensor to FP8 (Float8_e4m3fn) with scaling factor."""
    max_val = weight.abs().max().item()
    if max_val == 0.0:
        scale = 1.0
    else:
        # Float8_e4m3 max representable value is 448.0
        scale = max_val / 448.0

    scaled_w = weight / scale
    # Clamp to valid range [-448, 448]
    scaled_w = scaled_w.clamp(-448.0, 448.0)

    # Cast to Float8_e4m3fn if PyTorch supports it, else view as uint8
    if hasattr(torch, "float8_e4m3fn"):
        fp8_tensor = scaled_w.to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        # Fallback byte representation
        fp8_tensor = scaled_w.round().clamp(-128, 127).to(torch.int8).view(torch.uint8)

    return fp8_tensor, scale


def convert_linear_to_fp8(
    linear: nn.Linear,
    device: Optional[torch.device] = None
) -> QuantizedLinear:
    """Convert a standard nn.Linear layer into an FP8 QuantizedLinear layer."""
    dev = device or linear.weight.device
    qlinear = QuantizedLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        qtype=QuantType.FP8_E4M3,
        bias=linear.bias is not None,
        device=dev,
        dtype=linear.weight.dtype
    )

    qweight, scale_w = quantize_to_fp8_e4m3(linear.weight.data)
    qlinear.set_quant_weights(qweight, scale_w=scale_w)

    if linear.bias is not None:
        qlinear.bias.data.copy_(linear.bias.data)

    return qlinear
