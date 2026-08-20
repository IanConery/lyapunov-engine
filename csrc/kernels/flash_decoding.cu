#include "kernels/flash_decoding.cuh"
#include "utils/cuda_utils.cuh"
#include <cmath>
#include <cuda_fp16.h>

namespace lyapunov {
namespace kernels {

// ----------------------------------------------------------------------------
// Stage 1: Partitioned Split-KV Attention Kernel
// ----------------------------------------------------------------------------
template <typename T, int HEAD_DIM>
__global__ void flash_decoding_stage1_kernel(
    const T *__restrict__ q, const T *__restrict__ k, const T *__restrict__ v,
    T *__restrict__ mid_out, float *__restrict__ mid_lse, int num_heads,
    int num_kv_heads, int kv_seqlen, int num_partitions, int partition_size,
    int64_t q_batch_stride, int64_t q_head_stride, int64_t k_batch_stride,
    int64_t k_head_stride, int64_t k_seq_stride, int64_t v_batch_stride,
    int64_t v_head_stride, int64_t v_seq_stride, float sm_scale) {
  const int partition_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int batch_idx = blockIdx.z;

  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int kv_head_group_size = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / kv_head_group_size;

  const int seq_start = partition_idx * partition_size;
  const int seq_end = min(seq_start + partition_size, kv_seqlen);

  const T *q_ptr = q + batch_idx * q_batch_stride + head_idx * q_head_stride;
  const T *k_ptr = k + batch_idx * k_batch_stride + kv_head_idx * k_head_stride;
  const T *v_ptr = v + batch_idx * v_batch_stride + kv_head_idx * v_head_stride;

  // Load Q into shared memory
  __shared__ float s_q[HEAD_DIM];
  for (int d = tid; d < HEAD_DIM; d += num_threads) {
    s_q[d] = cuda::to_float(q_ptr[d]);
  }
  __syncthreads();

  float row_max = -1e20f;
  float row_sum = 0.0f;
  float acc[HEAD_DIM];

#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) {
    acc[d] = 0.0f;
  }

  // Process tokens in partition
  for (int t = seq_start; t < seq_end; ++t) {
    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
      float k_val = cuda::to_float(k_ptr[t * k_seq_stride + d]);
      score += s_q[d] * k_val;
    }
    score *= sm_scale;

    // Online softmax update
    float new_max = fmaxf(row_max, score);
    float scale_prev = expf(row_max - new_max);
    float p = expf(score - new_max);

    row_sum = row_sum * scale_prev + p;
    row_max = new_max;

    for (int d = 0; d < HEAD_DIM; ++d) {
      float v_val = cuda::to_float(v_ptr[t * v_seq_stride + d]);
      acc[d] = acc[d] * scale_prev + p * v_val;
    }
  }

  // Write partial outputs and LSE
  int mid_idx =
      (batch_idx * num_heads + head_idx) * num_partitions + partition_idx;

  if (row_sum > 0.0f) {
    float inv_sum = 1.0f / row_sum;
    for (int d = tid; d < HEAD_DIM; d += num_threads) {
      mid_out[mid_idx * HEAD_DIM + d] = cuda::from_float<T>(acc[d] * inv_sum);
    }
    if (tid == 0) {
      mid_lse[mid_idx] = row_max + logf(row_sum);
    }
  } else {
    for (int d = tid; d < HEAD_DIM; d += num_threads) {
      mid_out[mid_idx * HEAD_DIM + d] = cuda::from_float<T>(0.0f);
    }
    if (tid == 0) {
      mid_lse[mid_idx] = -1e20f;
    }
  }
}

// ----------------------------------------------------------------------------
// Stage 2: Reduction Merge Kernel across Partitions
// ----------------------------------------------------------------------------
template <typename T, int HEAD_DIM>
__global__ void flash_decoding_stage2_reduce_kernel(
    const T *__restrict__ mid_out, const float *__restrict__ mid_lse,
    T *__restrict__ out, int num_heads, int num_partitions,
    int64_t out_batch_stride, int64_t out_head_stride) {
  const int head_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int tid = threadIdx.x;

  const int base_mid_idx = (batch_idx * num_heads + head_idx) * num_partitions;
  T *out_ptr = out + batch_idx * out_batch_stride + head_idx * out_head_stride;

  // Find global max of partition LSEs
  float global_lse_max = -1e20f;
  for (int p = 0; p < num_partitions; ++p) {
    float lse_p = mid_lse[base_mid_idx + p];
    global_lse_max = fmaxf(global_lse_max, lse_p);
  }

  float weight_sum = 0.0f;
  float final_acc = 0.0f;

  if (tid < HEAD_DIM) {
    for (int p = 0; p < num_partitions; ++p) {
      float lse_p = mid_lse[base_mid_idx + p];
      if (lse_p > -1e10f) {
        float weight = expf(lse_p - global_lse_max);
        float val =
            cuda::to_float(mid_out[(base_mid_idx + p) * HEAD_DIM + tid]);
        final_acc += weight * val;
        if (tid == 0) {
          weight_sum += weight;
        }
      }
    }
  }

  __shared__ float s_weight_sum;
  if (tid == 0) {
    s_weight_sum = (weight_sum > 0.0f) ? (1.0f / weight_sum) : 0.0f;
  }
  __syncthreads();

  if (tid < HEAD_DIM) {
    out_ptr[tid] = cuda::from_float<T>(final_acc * s_weight_sum);
  }
}

void launch_flash_decoding(const FlashDecodingParams &params,
                           cudaStream_t stream) {
  int partition_size =
      (params.kv_seqlen + params.num_partitions - 1) / params.num_partitions;
  if (partition_size < 1)
    partition_size = 1;

  dim3 grid1(params.num_partitions, params.num_heads, params.batch_size);
  dim3 block1(128);

  dim3 grid2(params.num_heads, params.batch_size);
  dim3 block2(params.head_dim);

  if (params.dtype == CUDA_R_32F) {
    switch (params.head_dim) {
    case 64:
      flash_decoding_stage1_kernel<float, 64><<<grid1, block1, 0, stream>>>(
          static_cast<const float *>(params.q_ptr),
          static_cast<const float *>(params.k_ptr),
          static_cast<const float *>(params.v_ptr),
          static_cast<float *>(params.mid_out_ptr), params.mid_lse_ptr,
          params.num_heads, params.num_kv_heads, params.kv_seqlen,
          params.num_partitions, partition_size, params.q_batch_stride,
          params.q_head_stride, params.k_batch_stride, params.k_head_stride,
          params.k_seq_stride, params.v_batch_stride, params.v_head_stride,
          params.v_seq_stride, params.sm_scale);
      flash_decoding_stage2_reduce_kernel<float, 64>
          <<<grid2, block2, 0, stream>>>(
              static_cast<const float *>(params.mid_out_ptr),
              params.mid_lse_ptr, static_cast<float *>(params.out_ptr),
              params.num_heads, params.num_partitions, params.out_batch_stride,
              params.out_head_stride);
      break;
    case 128:
      flash_decoding_stage1_kernel<float, 128><<<grid1, block1, 0, stream>>>(
          static_cast<const float *>(params.q_ptr),
          static_cast<const float *>(params.k_ptr),
          static_cast<const float *>(params.v_ptr),
          static_cast<float *>(params.mid_out_ptr), params.mid_lse_ptr,
          params.num_heads, params.num_kv_heads, params.kv_seqlen,
          params.num_partitions, partition_size, params.q_batch_stride,
          params.q_head_stride, params.k_batch_stride, params.k_head_stride,
          params.k_seq_stride, params.v_batch_stride, params.v_head_stride,
          params.v_seq_stride, params.sm_scale);
      flash_decoding_stage2_reduce_kernel<float, 128>
          <<<grid2, block2, 0, stream>>>(
              static_cast<const float *>(params.mid_out_ptr),
              params.mid_lse_ptr, static_cast<float *>(params.out_ptr),
              params.num_heads, params.num_partitions, params.out_batch_stride,
              params.out_head_stride);
      break;
    default:
      throw std::invalid_argument(
          "Unsupported head dimension for FP32 flash_decoding");
    }
  } else if (params.dtype == CUDA_R_16F) {
    switch (params.head_dim) {
    case 64:
      flash_decoding_stage1_kernel<half, 64><<<grid1, block1, 0, stream>>>(
          static_cast<const half *>(params.q_ptr),
          static_cast<const half *>(params.k_ptr),
          static_cast<const half *>(params.v_ptr),
          static_cast<half *>(params.mid_out_ptr), params.mid_lse_ptr,
          params.num_heads, params.num_kv_heads, params.kv_seqlen,
          params.num_partitions, partition_size, params.q_batch_stride,
          params.q_head_stride, params.k_batch_stride, params.k_head_stride,
          params.k_seq_stride, params.v_batch_stride, params.v_head_stride,
          params.v_seq_stride, params.sm_scale);
      flash_decoding_stage2_reduce_kernel<half, 64>
          <<<grid2, block2, 0, stream>>>(
              static_cast<const half *>(params.mid_out_ptr), params.mid_lse_ptr,
              static_cast<half *>(params.out_ptr), params.num_heads,
              params.num_partitions, params.out_batch_stride,
              params.out_head_stride);
      break;
    case 128:
      flash_decoding_stage1_kernel<half, 128><<<grid1, block1, 0, stream>>>(
          static_cast<const half *>(params.q_ptr),
          static_cast<const half *>(params.k_ptr),
          static_cast<const half *>(params.v_ptr),
          static_cast<half *>(params.mid_out_ptr), params.mid_lse_ptr,
          params.num_heads, params.num_kv_heads, params.kv_seqlen,
          params.num_partitions, partition_size, params.q_batch_stride,
          params.q_head_stride, params.k_batch_stride, params.k_head_stride,
          params.k_seq_stride, params.v_batch_stride, params.v_head_stride,
          params.v_seq_stride, params.sm_scale);
      flash_decoding_stage2_reduce_kernel<half, 128>
          <<<grid2, block2, 0, stream>>>(
              static_cast<const half *>(params.mid_out_ptr), params.mid_lse_ptr,
              static_cast<half *>(params.out_ptr), params.num_heads,
              params.num_partitions, params.out_batch_stride,
              params.out_head_stride);
      break;
    default:
      throw std::invalid_argument(
          "Unsupported head dimension for FP16 flash_decoding");
    }
  } else {
    throw std::invalid_argument("Unsupported data type for flash_decoding");
  }

  CUDA_CHECK_LAST_ERROR();
}

} // namespace kernels
} // namespace lyapunov
