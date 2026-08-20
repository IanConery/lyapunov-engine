#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "utils/cuda_utils.cuh"

namespace lyapunov {
namespace kernels {

// Block definitions for GGUF quantization formats
#pragma pack(push, 1)

// Q4_0: 32 values per block (18 bytes total)
struct BlockQ4_0 {
    half d;              // 16-bit scale
    uint8_t qs[16];      // 32 nibbles (4 bits each)
};

// Q8_0: 32 values per block (34 bytes total)
struct BlockQ8_0 {
    half d;              // 16-bit scale
    int8_t qs[32];       // 32 8-bit signed integers
};

// Q4_K: 256 values per super-block (144 bytes total)
struct BlockQ4_K {
    half d;              // Super-block scale
    half dmin;           // Super-block min
    uint8_t scales[12];  // 6-bit sub-block scales and mins
    uint8_t qs[128];     // 256 4-bit nibbles
};

#pragma pack(pop)

// Quantization format enum
enum class QuantType {
    Q4_0 = 0,
    Q8_0 = 1,
    Q4_K = 2,
    FP8_E4M3 = 3,
    MARLIN_W4A16 = 4
};

// Host-callable launch functions
void launch_dequantize_q4_0(const void* src, float* dst, int num_elements, cudaStream_t stream = nullptr);
void launch_dequantize_q8_0(const void* src, float* dst, int num_elements, cudaStream_t stream = nullptr);
void launch_dequantize_q4_k(const void* src, float* dst, int num_elements, cudaStream_t stream = nullptr);

// Quantized GEMV: Y = X * W^T (where X is [batch, K], W is quantized [N, K], Y is [batch, N])
void launch_quant_gemv(
    const float* x,
    const void* qweight,
    const float* scales,
    float* y,
    int batch_size,
    int in_features,
    int out_features,
    QuantType qtype,
    cudaStream_t stream = nullptr
);

// FP8 Matrix Multiplication: Y = X * W^T with per-tensor scales
void launch_fp8_gemm(
    const void* x_fp8,
    const void* w_fp8,
    const float* scale_x,
    const float* scale_w,
    float* y,
    int m,
    int k,
    int n,
    cudaStream_t stream = nullptr
);

// Marlin W4A16 Matrix Multiplication
void launch_marlin_gemm(
    const float* x,
    const int32_t* qweight,
    const float* scales,
    float* y,
    int m,
    int k,
    int n,
    cudaStream_t stream = nullptr
);

} // namespace kernels
} // namespace lyapunov
