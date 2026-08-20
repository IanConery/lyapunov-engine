#include "kernels/quant_ops.cuh"
#include "utils/cuda_utils.cuh"
#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
#include <cuda_fp8.h>
#endif

namespace lyapunov {
namespace kernels {

// ----------------------------------------------------------------------------
// Dequantization Kernels
// ----------------------------------------------------------------------------

__global__ void dequantize_q4_0_kernel(
    const BlockQ4_0* __restrict__ src,
    float* __restrict__ dst,
    int num_blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const BlockQ4_0& b = src[block_idx];
    float d = cuda::to_float(b.d);
    float* out = dst + block_idx * 32;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint8_t byte = b.qs[i];
        int v0 = (byte & 0x0F) - 8;
        int v1 = (byte >> 4) - 8;
        out[i] = v0 * d;
        out[i + 16] = v1 * d;
    }
}

__global__ void dequantize_q8_0_kernel(
    const BlockQ8_0* __restrict__ src,
    float* __restrict__ dst,
    int num_blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const BlockQ8_0& b = src[block_idx];
    float d = cuda::to_float(b.d);
    float* out = dst + block_idx * 32;

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        out[i] = b.qs[i] * d;
    }
}

__global__ void dequantize_q4_k_kernel(
    const BlockQ4_K* __restrict__ src,
    float* __restrict__ dst,
    int num_super_blocks
) {
    int sb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sb_idx >= num_super_blocks) return;

    const BlockQ4_K& b = src[sb_idx];
    float d = cuda::to_float(b.d);
    float dmin = cuda::to_float(b.dmin);
    float* out = dst + sb_idx * 256;

    // Decode 8 sub-blocks of 32 elements
    for (int sb = 0; sb < 8; ++sb) {
        float sc = d * ((b.scales[sb] & 0x3F));
        float m = dmin * ((b.scales[sb] >> 6));
        const uint8_t* q_sub = b.qs + sb * 16;

        for (int i = 0; i < 16; ++i) {
            uint8_t byte = q_sub[i];
            out[sb * 32 + i] = (byte & 0x0F) * sc - m;
            out[sb * 32 + i + 16] = (byte >> 4) * sc - m;
        }
    }
}

// ----------------------------------------------------------------------------
// Quantized GEMV Kernel: Y = X * W^T (Single token / Batched decode)
// ----------------------------------------------------------------------------
template <int BLOCK_THREADS = 128>
__global__ void quant_gemv_q4_0_kernel(
    const float* __restrict__ x,
    const BlockQ4_0* __restrict__ w,
    float* __restrict__ y,
    int in_features,
    int out_features
) {
    int row = blockIdx.x; // Target output feature
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (row >= out_features) return;

    const float* x_vec = x + batch_idx * in_features;
    const BlockQ4_0* w_row = w + row * (in_features / 32);

    int num_blocks_per_row = in_features / 32;
    float sum = 0.0f;

    for (int b_idx = tid; b_idx < num_blocks_per_row; b_idx += BLOCK_THREADS) {
        const BlockQ4_0& b = w_row[b_idx];
        float d = cuda::to_float(b.d);
        const float* x_block = x_vec + b_idx * 32;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = b.qs[i];
            int v0 = (byte & 0x0F) - 8;
            int v1 = (byte >> 4) - 8;
            sum += v0 * d * x_block[i] + v1 * d * x_block[i + 16];
        }
    }

    __shared__ float s_scratch[WARP_SIZE];
    float total_sum = cuda::block_reduce_sum<float, BLOCK_THREADS>(sum, s_scratch);

    if (tid == 0) {
        y[batch_idx * out_features + row] = total_sum;
    }
}

template <int BLOCK_THREADS = 128>
__global__ void quant_gemv_q8_0_kernel(
    const float* __restrict__ x,
    const BlockQ8_0* __restrict__ w,
    float* __restrict__ y,
    int in_features,
    int out_features
) {
    int row = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (row >= out_features) return;

    const float* x_vec = x + batch_idx * in_features;
    const BlockQ8_0* w_row = w + row * (in_features / 32);

    int num_blocks_per_row = in_features / 32;
    float sum = 0.0f;

    for (int b_idx = tid; b_idx < num_blocks_per_row; b_idx += BLOCK_THREADS) {
        const BlockQ8_0& b = w_row[b_idx];
        float d = cuda::to_float(b.d);
        const float* x_block = x_vec + b_idx * 32;

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            sum += b.qs[i] * d * x_block[i];
        }
    }

    __shared__ float s_scratch[WARP_SIZE];
    float total_sum = cuda::block_reduce_sum<float, BLOCK_THREADS>(sum, s_scratch);

    if (tid == 0) {
        y[batch_idx * out_features + row] = total_sum;
    }
}

// ----------------------------------------------------------------------------
// FP8 GEMM / GEMV Kernel
// ----------------------------------------------------------------------------
template <int BLOCK_THREADS = 128>
__global__ void fp8_gemm_kernel(
    const uint8_t* __restrict__ x_fp8,
    const uint8_t* __restrict__ w_fp8,
    float scale_x,
    float scale_w,
    float* __restrict__ y,
    int in_features,
    int out_features
) {
    int row = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (row >= out_features) return;

    const uint8_t* x_vec = x_fp8 + batch_idx * in_features;
    const uint8_t* w_row = w_fp8 + row * in_features;
    float combined_scale = scale_x * scale_w;

    float sum = 0.0f;
    for (int k = tid; k < in_features; k += BLOCK_THREADS) {
        // Unpack FP8 E4M3 byte to float
        uint8_t bx = x_vec[k];
        uint8_t bw = w_row[k];

        // E4M3 sign, exp, mantissa unpack
        float fx = (bx & 0x80) ? -1.0f : 1.0f;
        int exp_x = (bx >> 3) & 0x0F;
        int frac_x = bx & 0x07;
        fx *= (exp_x == 0) ? (frac_x / 8.0f) * 0.125f : (1.0f + frac_x / 8.0f) * powf(2.0f, exp_x - 7);

        float fw = (bw & 0x80) ? -1.0f : 1.0f;
        int exp_w = (bw >> 3) & 0x0F;
        int frac_w = bw & 0x07;
        fw *= (exp_w == 0) ? (frac_w / 8.0f) * 0.125f : (1.0f + frac_w / 8.0f) * powf(2.0f, exp_w - 7);

        sum += fx * fw;
    }

    __shared__ float s_scratch[WARP_SIZE];
    float total_sum = cuda::block_reduce_sum<float, BLOCK_THREADS>(sum, s_scratch);

    if (tid == 0) {
        y[batch_idx * out_features + row] = total_sum * combined_scale;
    }
}

// ----------------------------------------------------------------------------
// Marlin-Style Packed W4A16 GEMV Kernel
// ----------------------------------------------------------------------------
template <int BLOCK_THREADS = 128>
__global__ void marlin_w4a16_gemv_kernel(
    const float* __restrict__ x,
    const int32_t* __restrict__ qweight,
    const float* __restrict__ scales,
    float* __restrict__ y,
    int in_features,
    int out_features
) {
    int row = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (row >= out_features) return;

    const float* x_vec = x + batch_idx * in_features;
    const int32_t* w_row = qweight + row * (in_features / 8);
    float scale = scales[row];

    float sum = 0.0f;
    int num_ints_per_row = in_features / 8;

    for (int i = tid; i < num_ints_per_row; i += BLOCK_THREADS) {
        int32_t packed = w_row[i];
        const float* x_sub = x_vec + i * 8;

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int val = ((packed >> (k * 4)) & 0x0F) - 8;
            sum += val * x_sub[k];
        }
    }

    __shared__ float s_scratch[WARP_SIZE];
    float total_sum = cuda::block_reduce_sum<float, BLOCK_THREADS>(sum, s_scratch);

    if (tid == 0) {
        y[batch_idx * out_features + row] = total_sum * scale;
    }
}

// ----------------------------------------------------------------------------
// Host Launch Dispatchers
// ----------------------------------------------------------------------------

void launch_dequantize_q4_0(const void* src, float* dst, int num_elements, cudaStream_t stream) {
    int num_blocks = num_elements / 32;
    int threads = 128;
    int grid = (num_blocks + threads - 1) / threads;
    dequantize_q4_0_kernel<<<grid, threads, 0, stream>>>(static_cast<const BlockQ4_0*>(src), dst, num_blocks);
    CUDA_CHECK_LAST_ERROR();
}

void launch_dequantize_q8_0(const void* src, float* dst, int num_elements, cudaStream_t stream) {
    int num_blocks = num_elements / 32;
    int threads = 128;
    int grid = (num_blocks + threads - 1) / threads;
    dequantize_q8_0_kernel<<<grid, threads, 0, stream>>>(static_cast<const BlockQ8_0*>(src), dst, num_blocks);
    CUDA_CHECK_LAST_ERROR();
}

void launch_dequantize_q4_k(const void* src, float* dst, int num_elements, cudaStream_t stream) {
    int num_sb = num_elements / 256;
    int threads = 128;
    int grid = (num_sb + threads - 1) / threads;
    dequantize_q4_k_kernel<<<grid, threads, 0, stream>>>(static_cast<const BlockQ4_K*>(src), dst, num_sb);
    CUDA_CHECK_LAST_ERROR();
}

void launch_quant_gemv(
    const float* x,
    const void* qweight,
    const float* scales,
    float* y,
    int batch_size,
    int in_features,
    int out_features,
    QuantType qtype,
    cudaStream_t stream
) {
    dim3 grid(out_features, batch_size);
    dim3 block(128);

    switch (qtype) {
        case QuantType::Q4_0:
            quant_gemv_q4_0_kernel<128><<<grid, block, 0, stream>>>(
                x, static_cast<const BlockQ4_0*>(qweight), y, in_features, out_features
            );
            break;
        case QuantType::Q8_0:
            quant_gemv_q8_0_kernel<128><<<grid, block, 0, stream>>>(
                x, static_cast<const BlockQ8_0*>(qweight), y, in_features, out_features
            );
            break;
        case QuantType::MARLIN_W4A16:
            marlin_w4a16_gemv_kernel<128><<<grid, block, 0, stream>>>(
                x, static_cast<const int32_t*>(qweight), scales, y, in_features, out_features
            );
            break;
        default:
            throw std::invalid_argument("Unsupported quantization format in launch_quant_gemv");
    }
    CUDA_CHECK_LAST_ERROR();
}

void launch_fp8_gemm(
    const void* x_fp8,
    const void* w_fp8,
    const float* scale_x,
    const float* scale_w,
    float* y,
    int m,
    int k,
    int n,
    cudaStream_t stream
) {
    dim3 grid(n, m);
    dim3 block(128);
    float sx = scale_x ? *scale_x : 1.0f;
    float sw = scale_w ? *scale_w : 1.0f;

    fp8_gemm_kernel<128><<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(x_fp8),
        static_cast<const uint8_t*>(w_fp8),
        sx, sw, y, k, n
    );
    CUDA_CHECK_LAST_ERROR();
}

void launch_marlin_gemm(
    const float* x,
    const int32_t* qweight,
    const float* scales,
    float* y,
    int m,
    int k,
    int n,
    cudaStream_t stream
) {
    dim3 grid(n, m);
    dim3 block(128);

    marlin_w4a16_gemv_kernel<128><<<grid, block, 0, stream>>>(
        x, qweight, scales, y, k, n
    );
    CUDA_CHECK_LAST_ERROR();
}

} // namespace kernels
} // namespace lyapunov
