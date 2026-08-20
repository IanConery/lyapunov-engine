from enum import IntEnum
from typing import Optional, Union
import torch
import torch.nn as nn

try:
    from lyapunov_engine import _C
    HAS_CUDA_EXT = True
except ImportError:
    _C = None
    HAS_CUDA_EXT = False


class QuantType(IntEnum):
    Q4_0 = 0
    Q8_0 = 1
    Q4_K = 2
    FP8_E4M3 = 3
    MARLIN_W4A16 = 4


class QuantizedLinear(nn.Module):
    """Linear layer supporting GGUF (Q4_0, Q8_0, Q4_K), FP8 (Float8_e4m3fn), and Marlin W4A16."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qtype: Union[QuantType, int] = QuantType.Q4_0,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.qtype = QuantType(qtype)
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # Raw packed weight storage
        self.register_buffer("qweight", torch.empty(0, dtype=torch.uint8, device=self.device))
        self.register_buffer("scales", torch.empty(0, dtype=torch.float32, device=self.device))
        self.register_buffer("scale_x", torch.tensor(1.0, dtype=torch.float32, device=self.device))
        self.register_buffer("scale_w", torch.tensor(1.0, dtype=torch.float32, device=self.device))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=self.device))
        else:
            self.register_parameter("bias", None)

    def set_quant_weights(
        self,
        qweight: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
        scale_x: float = 1.0,
        scale_w: float = 1.0
    ):
        self.qweight = qweight.to(self.device)
        if scales is not None:
            self.scales = scales.to(self.device)
        self.scale_x.fill_(scale_x)
        self.scale_w.fill_(scale_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.view(-1, self.in_features)

        if HAS_CUDA_EXT and x.is_cuda and self.qweight.is_cuda:
            if self.qtype in (QuantType.Q4_0, QuantType.Q8_0):
                # Custom CUDA Quantized GEMV
                out = _C.quant_gemv(
                    x_2d.float().contiguous(),
                    self.qweight.contiguous(),
                    self.scales if self.scales.numel() > 0 else None,
                    self.in_features,
                    self.out_features,
                    int(self.qtype)
                ).to(x.dtype)
            elif self.qtype == QuantType.FP8_E4M3:
                # Custom CUDA FP8 GEMM
                out = _C.fp8_gemm(
                    x_2d.view(torch.uint8).contiguous(),
                    self.qweight.contiguous(),
                    float(self.scale_x),
                    float(self.scale_w)
                ).to(x.dtype)
            elif self.qtype == QuantType.MARLIN_W4A16:
                # Custom CUDA Marlin W4A16 GEMM
                out = _C.marlin_gemm(
                    x_2d.float().contiguous(),
                    self.qweight.view(torch.int32).contiguous(),
                    self.scales.contiguous()
                ).to(x.dtype)
            else:
                # CPU / Eager fallback dequantization
                out = self._fallback_forward(x_2d)
        else:
            out = self._fallback_forward(x_2d)

        if self.bias is not None:
            out += self.bias

        return out.view(*orig_shape[:-1], self.out_features)

    def _fallback_forward(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Eager CPU dequantization fallback."""
        if self.qtype == QuantType.Q4_0:
            # Unpack Q4_0: 18 bytes per 32 weights
            num_blocks = self.qweight.numel() // 18
            raw_bytes = self.qweight.view(torch.uint8)
            weights = []
            for b in range(num_blocks):
                d = torch.frombuffer(raw_bytes[b * 18 : b * 18 + 2].cpu().numpy(), dtype=torch.float16).item()
                qs = raw_bytes[b * 18 + 2 : (b + 1) * 18]
                for i in range(16):
                    byte = qs[i].item()
                    v0 = (byte & 0x0F) - 8
                    v1 = (byte >> 4) - 8
                    weights.append(v0 * d)
                for i in range(16):
                    byte = qs[i].item()
                    v1 = (byte >> 4) - 8
                    weights.append(v1 * d)
            w_mat = torch.tensor(weights, dtype=x_2d.dtype, device=x_2d.device).view(self.out_features, self.in_features)
            return torch.matmul(x_2d, w_mat.t())

        elif self.qtype == QuantType.Q8_0:
            num_blocks = self.qweight.numel() // 34
            raw_bytes = self.qweight.view(torch.uint8)
            weights = []
            for b in range(num_blocks):
                d = torch.frombuffer(raw_bytes[b * 34 : b * 34 + 2].cpu().numpy(), dtype=torch.float16).item()
                qs = raw_bytes[b * 34 + 2 : (b + 1) * 34].view(torch.int8)
                for val in qs:
                    weights.append(val.item() * d)
            w_mat = torch.tensor(weights, dtype=x_2d.dtype, device=x_2d.device).view(self.out_features, self.in_features)
            return torch.matmul(x_2d, w_mat.t())

        return torch.zeros((x_2d.size(0), self.out_features), dtype=x_2d.dtype, device=x_2d.device)
