#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace lyapunov {
namespace kernels {

struct FlashAttnParams {
    void* q_ptr;
    void* k_ptr;
    void* v_ptr;
    void* out_ptr;
    float* lse_ptr; // Optional log-sum-exp output [batch_size, num_heads, q_seqlen]

    int batch_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int q_seqlen;
    int kv_seqlen;

    // Strides
    int64_t q_batch_stride;
    int64_t q_head_stride;
    int64_t q_seq_stride;

    int64_t k_batch_stride;
    int64_t k_head_stride;
    int64_t k_seq_stride;

    int64_t v_batch_stride;
    int64_t v_head_stride;
    int64_t v_seq_stride;

    int64_t out_batch_stride;
    int64_t out_head_stride;
    int64_t out_seq_stride;

    float sm_scale;
    bool is_causal;
    cudaDataType_t dtype;
};

void launch_flash_attn_v2(const FlashAttnParams& params, cudaStream_t stream = nullptr);

} // namespace kernels
} // namespace lyapunov
