#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace lyapunov {
namespace kernels {

struct FlashDecodingParams {
    void* q_ptr;              // [batch_size, num_heads, 1, head_dim]
    void* k_ptr;              // [batch_size, num_kv_heads, kv_seqlen, head_dim]
    void* v_ptr;              // [batch_size, num_kv_heads, kv_seqlen, head_dim]
    void* out_ptr;            // [batch_size, num_heads, 1, head_dim]
    
    void* mid_out_ptr;        // Workspace for partial outputs [batch_size, num_heads, num_partitions, head_dim]
    float* mid_lse_ptr;       // Workspace for partial LSE [batch_size, num_heads, num_partitions]

    int batch_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int kv_seqlen;
    int num_partitions;

    // Strides
    int64_t q_batch_stride;
    int64_t q_head_stride;

    int64_t k_batch_stride;
    int64_t k_head_stride;
    int64_t k_seq_stride;

    int64_t v_batch_stride;
    int64_t v_head_stride;
    int64_t v_seq_stride;

    int64_t out_batch_stride;
    int64_t out_head_stride;

    float sm_scale;
    cudaDataType_t dtype;
};

void launch_flash_decoding(const FlashDecodingParams& params, cudaStream_t stream = nullptr);

} // namespace kernels
} // namespace lyapunov
