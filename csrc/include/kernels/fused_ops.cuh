#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lyapunov {
namespace kernels {

// Fused RMSNorm with optional residual addition
// x: [num_tokens, hidden_dim]
// residual: [num_tokens, hidden_dim] (optional, can be nullptr)
// weight (gamma): [hidden_dim]
// out: [num_tokens, hidden_dim]
void launch_rmsnorm(void *out, void *out_residual, const void *input,
                    const void *residual, const void *weight, float eps,
                    int num_tokens, int hidden_dim, cudaDataType_t dtype,
                    cudaStream_t stream = nullptr);

// Fused SwiGLU: out = SiLU(gate) * up
// gate_up: [num_tokens, 2 * intermediate_dim] (gate in first half, up in second
// half or interleaved) out: [num_tokens, intermediate_dim]
void launch_swiglu(void *out, const void *gate_up, int num_tokens,
                   int intermediate_dim, cudaDataType_t dtype,
                   cudaStream_t stream = nullptr);

} // namespace kernels
} // namespace lyapunov
