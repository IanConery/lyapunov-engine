#pragma once

#include "utils/cuda_utils.cuh"
#include <cstdint>
#include <cuda_runtime.h>

namespace lyapunov {
namespace kernels {

struct FusedRoPEParams {
  void *q_out;   // [num_tokens, num_heads, head_dim]
  void *k_cache; // [num_blocks, num_kv_heads, block_size, head_dim]
  void *v_cache; // [num_blocks, num_kv_heads, block_size, head_dim]

  const void *q_in; // [num_tokens, num_heads, head_dim]
  const void *k_in; // [num_tokens, num_kv_heads, head_dim]
  const void *v_in; // [num_tokens, num_kv_heads, head_dim]

  const float *cos_sin_cache;  // [max_position, head_dim]
  const int32_t *block_tables; // [num_seqs, max_blocks_per_seq]
  const int32_t *context_lens; // [num_seqs]
  const int32_t *seq_idx_map;  // [num_tokens]

  int num_tokens;
  int num_heads;
  int num_kv_heads;
  int head_dim;
  int block_size;
  int max_blocks_per_seq;
  cudaDataType_t dtype;
};

void launch_fused_rope_paged(const FusedRoPEParams &params,
                             cudaStream_t stream = nullptr);

} // namespace kernels
} // namespace lyapunov
