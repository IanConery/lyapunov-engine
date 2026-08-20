#include "kernels/flash_attn.cuh"
#include "utils/cuda_utils.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>

namespace lyapunov {
namespace kernels {

constexpr int BLOCK_M = 32; // Query tile size
constexpr int BLOCK_N = 32; // Key/Value tile size

template <typename T, int HEAD_DIM>
__global__ void flash_attn_v2_fwd_kernel(
    const T *__restrict__ q, const T *__restrict__ k, const T *__restrict__ v,
    T *__restrict__ out, float *__restrict__ lse, int batch_size, int num_heads,
    int num_kv_heads, int q_seqlen, int kv_seqlen, int64_t q_batch_stride,
    int64_t q_head_stride, int64_t q_seq_stride, int64_t k_batch_stride,
    int64_t k_head_stride, int64_t k_seq_stride, int64_t v_batch_stride,
    int64_t v_head_stride, int64_t v_seq_stride, int64_t out_batch_stride,
    int64_t out_head_stride, int64_t out_seq_stride, float sm_scale,
    bool is_causal) {
  const int q_block_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int batch_idx = blockIdx.z;

  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int kv_head_group_size = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / kv_head_group_size;

  const int q_start = q_block_idx * BLOCK_M;
  if (q_start >= q_seqlen)
    return;

  const T *q_ptr = q + batch_idx * q_batch_stride + head_idx * q_head_stride;
  const T *k_ptr = k + batch_idx * k_batch_stride + kv_head_idx * k_head_stride;
  const T *v_ptr = v + batch_idx * v_batch_stride + kv_head_idx * v_head_stride;
  T *out_ptr = out + batch_idx * out_batch_stride + head_idx * out_head_stride;
  float *lse_ptr =
      lse ? (lse + (batch_idx * num_heads + head_idx) * q_seqlen) : nullptr;

  extern __shared__ char smem_buffer[];
  T *s_q = reinterpret_cast<T *>(smem_buffer);
  T *s_k = s_q + BLOCK_M * HEAD_DIM;
  T *s_v = s_k + BLOCK_N * HEAD_DIM;
  float *s_p =
      reinterpret_cast<float *>(s_v + BLOCK_N * HEAD_DIM); // [BLOCK_M, BLOCK_N]

  __shared__ float s_row_max[BLOCK_M];
  __shared__ float s_row_sum[BLOCK_M];
  __shared__ float s_scale_prev[BLOCK_M];

  if (tid < BLOCK_M) {
    s_row_max[tid] = -1e20f;
    s_row_sum[tid] = 0.0f;
  }

  // Thread-local accumulation buffer for this block's Q rows
  float acc_out[BLOCK_M * HEAD_DIM / 128 > 0 ? BLOCK_M * HEAD_DIM / 128 : 1];
  const int elems_per_thread =
      (BLOCK_M * HEAD_DIM + num_threads - 1) / num_threads;
  float thread_acc[32]; // Max 32 elements per thread for 32x128 / 128 threads =
                        // 32
  for (int i = 0; i < elems_per_thread; ++i) {
    thread_acc[i] = 0.0f;
  }

  // Load Q block into Shared Memory
  for (int idx = tid; idx < BLOCK_M * HEAD_DIM; idx += num_threads) {
    int m = idx / HEAD_DIM;
    int d = idx % HEAD_DIM;
    int global_m = q_start + m;
    if (global_m < q_seqlen) {
      s_q[idx] = q_ptr[global_m * q_seq_stride + d];
    } else {
      s_q[idx] = cuda::from_float<T>(0.0f);
    }
  }
  __syncthreads();

  const int num_kv_blocks = (kv_seqlen + BLOCK_N - 1) / BLOCK_N;

  for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
    const int kv_start = kv_block_idx * BLOCK_N;
    if (is_causal && kv_start > (q_start + BLOCK_M - 1)) {
      break;
    }

    // Load K and V tiles into Shared Memory
    for (int idx = tid; idx < BLOCK_N * HEAD_DIM; idx += num_threads) {
      int n = idx / HEAD_DIM;
      int d = idx % HEAD_DIM;
      int global_n = kv_start + n;
      if (global_n < kv_seqlen) {
        s_k[idx] = k_ptr[global_n * k_seq_stride + d];
        s_v[idx] = v_ptr[global_n * v_seq_stride + d];
      } else {
        s_k[idx] = cuda::from_float<T>(0.0f);
        s_v[idx] = cuda::from_float<T>(0.0f);
      }
    }
    __syncthreads();

    // 1. Compute S = Q * K^T * sm_scale into s_p
    for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += num_threads) {
      int m = idx / BLOCK_N;
      int n = idx % BLOCK_N;
      int global_m = q_start + m;
      int global_n = kv_start + n;

      if (global_m < q_seqlen && global_n < kv_seqlen) {
        if (is_causal && global_n > global_m) {
          s_p[idx] = -1e20f;
        } else {
          float score = 0.0f;
#pragma unroll
          for (int d = 0; d < HEAD_DIM; ++d) {
            score += cuda::to_float(s_q[m * HEAD_DIM + d]) *
                     cuda::to_float(s_k[n * HEAD_DIM + d]);
          }
          s_p[idx] = score * sm_scale;
        }
      } else {
        s_p[idx] = -1e20f;
      }
    }
    __syncthreads();

    // 2. Row-wise online softmax stats (each thread in warp handles one row m)
    if (tid < BLOCK_M) {
      int m = tid;
      int global_m = q_start + m;
      if (global_m < q_seqlen) {
        float block_max = -1e20f;
#pragma unroll
        for (int n = 0; n < BLOCK_N; ++n) {
          block_max = fmaxf(block_max, s_p[m * BLOCK_N + n]);
        }

        if (block_max > -1e10f) {
          float new_max = fmaxf(s_row_max[m], block_max);
          float scale_prev = expf(s_row_max[m] - new_max);
          float scale_curr = expf(block_max - new_max);

          float block_sum = 0.0f;
#pragma unroll
          for (int n = 0; n < BLOCK_N; ++n) {
            float p_val = expf(s_p[m * BLOCK_N + n] - block_max) * scale_curr;
            s_p[m * BLOCK_N + n] = p_val;
            block_sum += p_val;
          }

          s_scale_prev[m] = scale_prev;
          s_row_sum[m] = s_row_sum[m] * scale_prev + block_sum;
          s_row_max[m] = new_max;
        } else {
          s_scale_prev[m] = 1.0f;
#pragma unroll
          for (int n = 0; n < BLOCK_N; ++n) {
            s_p[m * BLOCK_N + n] = 0.0f;
          }
        }
      }
    }
    __syncthreads();

    // 3. Parallel matrix multiply: O = O * scale_prev + P * V
    for (int i = 0; i < elems_per_thread; ++i) {
      int idx = tid * elems_per_thread + i;
      if (idx < BLOCK_M * HEAD_DIM) {
        int m = idx / HEAD_DIM;
        int d = idx % HEAD_DIM;

        float pv = 0.0f;
#pragma unroll
        for (int n = 0; n < BLOCK_N; ++n) {
          pv += s_p[m * BLOCK_N + n] * cuda::to_float(s_v[n * HEAD_DIM + d]);
        }

        thread_acc[i] = thread_acc[i] * s_scale_prev[m] + pv;
      }
    }
    __syncthreads();
  }

  // Write final normalized outputs to global memory
  for (int i = 0; i < elems_per_thread; ++i) {
    int idx = tid * elems_per_thread + i;
    if (idx < BLOCK_M * HEAD_DIM) {
      int m = idx / HEAD_DIM;
      int d = idx % HEAD_DIM;
      int global_m = q_start + m;

      if (global_m < q_seqlen) {
        float inv_sum = (s_row_sum[m] > 0.0f) ? (1.0f / s_row_sum[m]) : 0.0f;
        out_ptr[global_m * out_seq_stride + d] =
            cuda::from_float<T>(thread_acc[i] * inv_sum);
      }
    }
  }

  if (lse_ptr && tid < BLOCK_M) {
    int global_m = q_start + tid;
    if (global_m < q_seqlen) {
      lse_ptr[global_m] = (s_row_sum[tid] > 0.0f)
                              ? (s_row_max[tid] + logf(s_row_sum[tid]))
                              : -1e20f;
    }
  }
}

void launch_flash_attn_v2(const FlashAttnParams &params, cudaStream_t stream) {
  const int num_q_blocks = (params.q_seqlen + BLOCK_M - 1) / BLOCK_M;
  dim3 grid(num_q_blocks, params.num_heads, params.batch_size);
  dim3 block(128);

  size_t element_size =
      (params.dtype == CUDA_R_32F) ? sizeof(float) : sizeof(half);
  size_t smem_bytes =
      (BLOCK_M * params.head_dim + 2 * BLOCK_N * params.head_dim) *
          element_size +
      (BLOCK_M * BLOCK_N) * sizeof(float);

  if (params.dtype == CUDA_R_32F) {
    switch (params.head_dim) {
    case 64:
      flash_attn_v2_fwd_kernel<float, 64><<<grid, block, smem_bytes, stream>>>(
          static_cast<const float *>(params.q_ptr),
          static_cast<const float *>(params.k_ptr),
          static_cast<const float *>(params.v_ptr),
          static_cast<float *>(params.out_ptr), params.lse_ptr,
          params.batch_size, params.num_heads, params.num_kv_heads,
          params.q_seqlen, params.kv_seqlen, params.q_batch_stride,
          params.q_head_stride, params.q_seq_stride, params.k_batch_stride,
          params.k_head_stride, params.k_seq_stride, params.v_batch_stride,
          params.v_head_stride, params.v_seq_stride, params.out_batch_stride,
          params.out_head_stride, params.out_seq_stride, params.sm_scale,
          params.is_causal);
      break;
    case 128:
      if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(flash_attn_v2_fwd_kernel<float, 128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
      }
      flash_attn_v2_fwd_kernel<float, 128><<<grid, block, smem_bytes, stream>>>(
          static_cast<const float *>(params.q_ptr),
          static_cast<const float *>(params.k_ptr),
          static_cast<const float *>(params.v_ptr),
          static_cast<float *>(params.out_ptr), params.lse_ptr,
          params.batch_size, params.num_heads, params.num_kv_heads,
          params.q_seqlen, params.kv_seqlen, params.q_batch_stride,
          params.q_head_stride, params.q_seq_stride, params.k_batch_stride,
          params.k_head_stride, params.k_seq_stride, params.v_batch_stride,
          params.v_head_stride, params.v_seq_stride, params.out_batch_stride,
          params.out_head_stride, params.out_seq_stride, params.sm_scale,
          params.is_causal);
      break;
    default:
      throw std::invalid_argument(
          "Unsupported head dimension for FP32 flash_attn_v2");
    }
  } else if (params.dtype == CUDA_R_16F) {
    switch (params.head_dim) {
    case 64:
      flash_attn_v2_fwd_kernel<half, 64><<<grid, block, smem_bytes, stream>>>(
          static_cast<const half *>(params.q_ptr),
          static_cast<const half *>(params.k_ptr),
          static_cast<const half *>(params.v_ptr),
          static_cast<half *>(params.out_ptr), params.lse_ptr,
          params.batch_size, params.num_heads, params.num_kv_heads,
          params.q_seqlen, params.kv_seqlen, params.q_batch_stride,
          params.q_head_stride, params.q_seq_stride, params.k_batch_stride,
          params.k_head_stride, params.k_seq_stride, params.v_batch_stride,
          params.v_head_stride, params.v_seq_stride, params.out_batch_stride,
          params.out_head_stride, params.out_seq_stride, params.sm_scale,
          params.is_causal);
      break;
    case 128:
      flash_attn_v2_fwd_kernel<half, 128><<<grid, block, smem_bytes, stream>>>(
          static_cast<const half *>(params.q_ptr),
          static_cast<const half *>(params.k_ptr),
          static_cast<const half *>(params.v_ptr),
          static_cast<half *>(params.out_ptr), params.lse_ptr,
          params.batch_size, params.num_heads, params.num_kv_heads,
          params.q_seqlen, params.kv_seqlen, params.q_batch_stride,
          params.q_head_stride, params.q_seq_stride, params.k_batch_stride,
          params.k_head_stride, params.k_seq_stride, params.v_batch_stride,
          params.v_head_stride, params.v_seq_stride, params.out_batch_stride,
          params.out_head_stride, params.out_seq_stride, params.sm_scale,
          params.is_causal);
      break;
    default:
      throw std::invalid_argument(
          "Unsupported head dimension for FP16 flash_attn_v2");
    }
  } else {
    throw std::invalid_argument("Unsupported data type for flash_attn_v2");
  }

  CUDA_CHECK_LAST_ERROR();
}

} // namespace kernels
} // namespace lyapunov
