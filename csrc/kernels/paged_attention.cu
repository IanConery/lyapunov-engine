#include "kernels/paged_attn.cuh"
#include "utils/cuda_utils.cuh"
#include <cmath>
#include <cuda_fp16.h>

namespace lyapunov {
namespace kernels {

template <typename T, int HEAD_DIM, int BLOCK_SIZE = 16>
__global__ void paged_attention_kernel(
    T *__restrict__ out, const T *__restrict__ q, const T *__restrict__ k_cache,
    const T *__restrict__ v_cache, const int32_t *__restrict__ block_tables,
    const int32_t *__restrict__ context_lens, int num_heads, int num_kv_heads,
    int max_num_blocks_per_seq, int64_t q_batch_stride, int64_t q_head_stride,
    int64_t k_block_stride, int64_t k_head_stride, int64_t k_token_stride,
    int64_t v_block_stride, int64_t v_head_stride, int64_t v_token_stride,
    int64_t out_batch_stride, int64_t out_head_stride, float sm_scale) {
  // Grid: (num_heads, batch_size)
  const int head_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;

  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int context_len = context_lens[batch_idx];
  if (context_len <= 0)
    return;

  const int kv_head_group_size = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / kv_head_group_size;

  const T *q_ptr = q + batch_idx * q_batch_stride + head_idx * q_head_stride;
  T *out_ptr = out + batch_idx * out_batch_stride + head_idx * out_head_stride;
  const int32_t *block_table =
      block_tables + batch_idx * max_num_blocks_per_seq;

  // Load Q into shared memory
  __shared__ float s_q[HEAD_DIM];
  for (int d = tid; d < HEAD_DIM; d += num_threads) {
    s_q[d] = cuda::to_float(q_ptr[d]);
  }
  __syncthreads();

  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  float row_max = -1e20f;
  float row_sum = 0.0f;
  float acc[HEAD_DIM];

#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) {
    acc[d] = 0.0f;
  }

  // Shared scratch space for block reduction
  __shared__ float s_scratch[WARP_SIZE];

  // Iterate over physical KV blocks
  for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
    const int physical_block_idx = block_table[block_idx];
    const int tokens_in_block =
        min(BLOCK_SIZE, context_len - block_idx * BLOCK_SIZE);

    const T *k_block_ptr = k_cache + physical_block_idx * k_block_stride +
                           kv_head_idx * k_head_stride;
    const T *v_block_ptr = v_cache + physical_block_idx * v_block_stride +
                           kv_head_idx * v_head_stride;

    for (int token_idx = 0; token_idx < tokens_in_block; ++token_idx) {
      const T *k_tok = k_block_ptr + token_idx * k_token_stride;
      const T *v_tok = v_block_ptr + token_idx * v_token_stride;

      // Compute dot product S = Q * K
      float score = 0.0f;
      for (int d = tid; d < HEAD_DIM; d += num_threads) {
        score += s_q[d] * cuda::to_float(k_tok[d]);
      }
      score = cuda::block_reduce_sum<float, 128>(score, s_scratch);

      __shared__ float s_token_score;
      if (tid == 0) {
        s_token_score = score * sm_scale;
      }
      __syncthreads();

      float tok_score = s_token_score;

      // Online softmax update
      float new_max = fmaxf(row_max, tok_score);
      float scale_prev = expf(row_max - new_max);
      float p = expf(tok_score - new_max);

      row_sum = row_sum * scale_prev + p;
      row_max = new_max;

      for (int d = tid; d < HEAD_DIM; d += num_threads) {
        float v_val = cuda::to_float(v_tok[d]);
        acc[d] = acc[d] * scale_prev + p * v_val;
      }
      __syncthreads();
    }
  }

  // Store final normalized output
  float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
  for (int d = tid; d < HEAD_DIM; d += num_threads) {
    out_ptr[d] = cuda::from_float<T>(acc[d] * inv_sum);
  }
}

void launch_paged_attention(const PagedAttentionParams &params,
                            cudaStream_t stream) {
  dim3 grid(params.num_heads, params.batch_size);
  dim3 block(128);

  if (params.dtype == CUDA_R_32F) {
    switch (params.head_dim) {
    case 64:
      paged_attention_kernel<float, 64><<<grid, block, 0, stream>>>(
          static_cast<float *>(params.out_ptr),
          static_cast<const float *>(params.q_ptr),
          static_cast<const float *>(params.k_cache_ptr),
          static_cast<const float *>(params.v_cache_ptr), params.block_tables,
          params.context_lens, params.num_heads, params.num_kv_heads,
          params.max_num_blocks_per_seq, params.q_batch_stride,
          params.q_head_stride, params.k_block_stride, params.k_head_stride,
          params.k_token_stride, params.v_block_stride, params.v_head_stride,
          params.v_token_stride, params.out_batch_stride,
          params.out_head_stride, params.sm_scale);
      break;
    case 128:
      paged_attention_kernel<float, 128><<<grid, block, 0, stream>>>(
          static_cast<float *>(params.out_ptr),
          static_cast<const float *>(params.q_ptr),
          static_cast<const float *>(params.k_cache_ptr),
          static_cast<const float *>(params.v_cache_ptr), params.block_tables,
          params.context_lens, params.num_heads, params.num_kv_heads,
          params.max_num_blocks_per_seq, params.q_batch_stride,
          params.q_head_stride, params.k_block_stride, params.k_head_stride,
          params.k_token_stride, params.v_block_stride, params.v_head_stride,
          params.v_token_stride, params.out_batch_stride,
          params.out_head_stride, params.sm_scale);
      break;
    default:
      throw std::invalid_argument(
          "Unsupported head dimension for FP32 paged_attention");
    }
  } else if (params.dtype == CUDA_R_16F) {
    switch (params.head_dim) {
    case 64:
      paged_attention_kernel<half, 64><<<grid, block, 0, stream>>>(
          static_cast<half *>(params.out_ptr),
          static_cast<const half *>(params.q_ptr),
          static_cast<const half *>(params.k_cache_ptr),
          static_cast<const half *>(params.v_cache_ptr), params.block_tables,
          params.context_lens, params.num_heads, params.num_kv_heads,
          params.max_num_blocks_per_seq, params.q_batch_stride,
          params.q_head_stride, params.k_block_stride, params.k_head_stride,
          params.k_token_stride, params.v_block_stride, params.v_head_stride,
          params.v_token_stride, params.out_batch_stride,
          params.out_head_stride, params.sm_scale);
      break;
    case 128:
      paged_attention_kernel<half, 128><<<grid, block, 0, stream>>>(
          static_cast<half *>(params.out_ptr),
          static_cast<const half *>(params.q_ptr),
          static_cast<const half *>(params.k_cache_ptr),
          static_cast<const half *>(params.v_cache_ptr), params.block_tables,
          params.context_lens, params.num_heads, params.num_kv_heads,
          params.max_num_blocks_per_seq, params.q_batch_stride,
          params.q_head_stride, params.k_block_stride, params.k_head_stride,
          params.k_token_stride, params.v_block_stride, params.v_head_stride,
          params.v_token_stride, params.out_batch_stride,
          params.out_head_stride, params.sm_scale);
      break;
    default:
      throw std::invalid_argument(
          "Unsupported head dimension for FP16 paged_attention");
    }
  } else {
    throw std::invalid_argument("Unsupported data type for paged_attention");
  }

  CUDA_CHECK_LAST_ERROR();
}

} // namespace kernels
} // namespace lyapunov
