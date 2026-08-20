#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lyapunov {
namespace kernels {

struct PagedAttentionParams {
  void *out_ptr;           // [batch_size, num_heads, head_dim]
  const void *q_ptr;       // [batch_size, num_heads, head_dim]
  const void *k_cache_ptr; // [num_blocks, num_kv_heads, block_size, head_dim]
  const void *v_cache_ptr; // [num_blocks, num_kv_heads, block_size, head_dim]
  const int32_t *block_tables; // [batch_size, max_num_blocks_per_seq]
  const int32_t *context_lens; // [batch_size]

  int batch_size;
  int num_heads;
  int num_kv_heads;
  int head_dim;
  int block_size;
  int max_num_blocks_per_seq;

  // Strides
  int64_t q_batch_stride;
  int64_t q_head_stride;

  int64_t k_block_stride;
  int64_t k_head_stride;
  int64_t k_token_stride;

  int64_t v_block_stride;
  int64_t v_head_stride;
  int64_t v_token_stride;

  int64_t out_batch_stride;
  int64_t out_head_stride;

  float sm_scale;
  cudaDataType_t dtype;
};

void launch_paged_attention(const PagedAttentionParams &params,
                            cudaStream_t stream = nullptr);

} // namespace kernels
} // namespace lyapunov
