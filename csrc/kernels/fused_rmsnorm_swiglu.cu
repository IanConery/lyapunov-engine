#include "kernels/fused_ops.cuh"
#include "utils/cuda_utils.cuh"
#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

namespace lyapunov {
namespace kernels {

// ----------------------------------------------------------------------------
// RMSNorm Kernels
// ----------------------------------------------------------------------------

template <typename T, int VEC_SIZE = 4>
__global__ void
rmsnorm_kernel(T *__restrict__ out, T *__restrict__ out_residual,
               const T *__restrict__ input, const T *__restrict__ residual,
               const T *__restrict__ weight, float eps, int hidden_dim) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int row_offset = token_idx * hidden_dim;
  const T *in_row = input + row_offset;
  const T *res_row = residual ? (residual + row_offset) : nullptr;
  T *out_res_row = out_residual ? (out_residual + row_offset) : nullptr;
  T *out_row = out + row_offset;

  __shared__ float s_variance;
  __shared__ float s_scratch[32];

  float sum_sq = 0.0f;

  // Vectorized accumulation of sum(x^2)
  const int vec_elements = hidden_dim / VEC_SIZE;
  for (int i = tid; i < vec_elements; i += num_threads) {
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      int elem_idx = i * VEC_SIZE + v;
      float val = cuda::to_float(in_row[elem_idx]);
      if (res_row != nullptr) {
        float res_val = cuda::to_float(res_row[elem_idx]);
        val += res_val;
        if (out_res_row != nullptr) {
          out_res_row[elem_idx] = cuda::from_float<T>(val);
        }
      }
      sum_sq += val * val;
    }
  }

  // Process remaining tail elements if hidden_dim is not divisible by VEC_SIZE
  const int tail_start = vec_elements * VEC_SIZE;
  for (int i = tail_start + tid; i < hidden_dim; i += num_threads) {
    float val = cuda::to_float(in_row[i]);
    if (res_row != nullptr) {
      float res_val = cuda::to_float(res_row[i]);
      val += res_val;
      if (out_res_row != nullptr) {
        out_res_row[i] = cuda::from_float<T>(val);
      }
    }
    sum_sq += val * val;
  }

  // Warp and Block level reduction
  sum_sq = cuda::warp_reduce_sum(sum_sq);

  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;

  if (lane_id == 0) {
    s_scratch[warp_id] = sum_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_sum = (lane_id < num_warps) ? s_scratch[lane_id] : 0.0f;
    block_sum = cuda::warp_reduce_sum(block_sum);
    if (lane_id == 0) {
      float mean_sq = block_sum / static_cast<float>(hidden_dim);
      s_variance = rsqrtf(mean_sq + eps);
    }
  }
  __syncthreads();

  const float inv_rms = s_variance;

  // Normalize and scale by gamma (weight)
  for (int i = tid; i < vec_elements; i += num_threads) {
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      int elem_idx = i * VEC_SIZE + v;
      float val = cuda::to_float(in_row[elem_idx]);
      if (res_row != nullptr) {
        val += cuda::to_float(res_row[elem_idx]);
      }
      float gamma = cuda::to_float(weight[elem_idx]);
      out_row[elem_idx] = cuda::from_float<T>(val * inv_rms * gamma);
    }
  }

  for (int i = tail_start + tid; i < hidden_dim; i += num_threads) {
    float val = cuda::to_float(in_row[i]);
    if (res_row != nullptr) {
      val += cuda::to_float(res_row[i]);
    }
    float gamma = cuda::to_float(weight[i]);
    out_row[i] = cuda::from_float<T>(val * inv_rms * gamma);
  }
}

void launch_rmsnorm(void *out, void *out_residual, const void *input,
                    const void *residual, const void *weight, float eps,
                    int num_tokens, int hidden_dim, cudaDataType_t dtype,
                    cudaStream_t stream) {
  if (num_tokens == 0 || hidden_dim == 0)
    return;

  int threads_per_block = 256;
  if (hidden_dim <= 512)
    threads_per_block = 128;
  else if (hidden_dim >= 4096)
    threads_per_block = 512;

  dim3 grid(num_tokens);
  dim3 block(threads_per_block);

  if (dtype == CUDA_R_32F) {
    rmsnorm_kernel<float><<<grid, block, 0, stream>>>(
        static_cast<float *>(out), static_cast<float *>(out_residual),
        static_cast<const float *>(input), static_cast<const float *>(residual),
        static_cast<const float *>(weight), eps, hidden_dim);
  } else if (dtype == CUDA_R_16F) {
    rmsnorm_kernel<half><<<grid, block, 0, stream>>>(
        static_cast<half *>(out), static_cast<half *>(out_residual),
        static_cast<const half *>(input), static_cast<const half *>(residual),
        static_cast<const half *>(weight), eps, hidden_dim);
  } else {
    throw std::invalid_argument("Unsupported data type for launch_rmsnorm");
  }

  CUDA_CHECK_LAST_ERROR();
}

// ----------------------------------------------------------------------------
// SwiGLU Kernels
// ----------------------------------------------------------------------------

template <typename T, int VEC_SIZE = 4>
__global__ void swiglu_kernel(T *__restrict__ out,
                              const T *__restrict__ gate_up,
                              int intermediate_dim, int total_elements) {
  const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
  if (idx >= total_elements)
    return;

#pragma unroll
  for (int v = 0; v < VEC_SIZE; ++v) {
    int cur_idx = idx + v;
    if (cur_idx < total_elements) {
      int token_idx = cur_idx / intermediate_dim;
      int dim_idx = cur_idx % intermediate_dim;

      int gate_offset = token_idx * (2 * intermediate_dim) + dim_idx;
      int up_offset = gate_offset + intermediate_dim;

      float gate = cuda::to_float(gate_up[gate_offset]);
      float up = cuda::to_float(gate_up[up_offset]);

      float silu_gate = gate / (1.0f + __expf(-gate));
      float result = silu_gate * up;

      out[cur_idx] = cuda::from_float<T>(result);
    }
  }
}

void launch_swiglu(void *out, const void *gate_up, int num_tokens,
                   int intermediate_dim, cudaDataType_t dtype,
                   cudaStream_t stream) {
  int total_elements = num_tokens * intermediate_dim;
  if (total_elements == 0)
    return;

  constexpr int VEC_SIZE = 4;
  int num_vec_elements = (total_elements + VEC_SIZE - 1) / VEC_SIZE;
  int threads_per_block = 256;
  int blocks = (num_vec_elements + threads_per_block - 1) / threads_per_block;

  if (dtype == CUDA_R_32F) {
    swiglu_kernel<float, VEC_SIZE><<<blocks, threads_per_block, 0, stream>>>(
        static_cast<float *>(out), static_cast<const float *>(gate_up),
        intermediate_dim, total_elements);
  } else if (dtype == CUDA_R_16F) {
    swiglu_kernel<half, VEC_SIZE><<<blocks, threads_per_block, 0, stream>>>(
        static_cast<half *>(out), static_cast<const half *>(gate_up),
        intermediate_dim, total_elements);
  } else {
    throw std::invalid_argument("Unsupported data type for launch_swiglu");
  }

  CUDA_CHECK_LAST_ERROR();
}

} // namespace kernels
} // namespace lyapunov
