import torch
from torch import nn

from lyapunov_engine.models.quant import QuantizedLinear, QuantType


def quantize_and_pack_marlin_w4a16(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an FP16/FP32 matrix to 4-bit signed integers and pack 8 nibbles per int32."""
    out_features, in_features = weight.shape
    assert in_features % 8 == 0, "in_features must be a multiple of 8"

    # Compute per-row scale
    max_val = weight.abs().max(dim=1, keepdim=True).values.clamp(min=1e-5)
    scale = (max_val / 7.0).squeeze(-1)  # 4-bit signed max magnitude is 7

    # Quantize to [-8, 7]
    q_unpacked = (weight / scale.unsqueeze(1)).round().clamp(-8, 7).to(torch.int32)
    # Map signed [-8, 7] to unsigned [0, 15] by adding 8
    q_unsigned = (q_unpacked + 8).to(torch.int32)

    # Pack 8 nibbles into one int32
    num_ints = in_features // 8
    packed = torch.zeros(
        (out_features, num_ints), dtype=torch.int32, device=weight.device
    )

    for k in range(8):
        nibble = q_unsigned[:, k::8]
        packed |= nibble << (k * 4)

    return packed, scale.float()


def convert_linear_to_marlin_w4a16(
    linear: nn.Linear, device: torch.device | None = None
) -> QuantizedLinear:
    """Convert a standard nn.Linear layer into a Marlin W4A16 QuantizedLinear layer."""
    dev = device or linear.weight.device
    qlinear = QuantizedLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        qtype=QuantType.MARLIN_W4A16,
        bias=linear.bias is not None,
        device=dev,
        dtype=linear.weight.dtype,
    )

    qweight, scales = quantize_and_pack_marlin_w4a16(linear.weight.data)
    qlinear.set_quant_weights(qweight, scales=scales)

    if linear.bias is not None:
        qlinear.bias.data.copy_(linear.bias.data)

    return qlinear
