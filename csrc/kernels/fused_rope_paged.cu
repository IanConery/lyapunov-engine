#include "kernels/fused_rope.cuh"
#include "utils/cuda_utils.cuh"

namespace lyapunov {
namespace kernels {

template <typename T>
__global__ void fused_rope_paged_kernel(
    T* __restrict__ q_out,
    T* __restrict__ k_cache,
    T* __restrict__ v_cache,
    const T* __restrict__ q_in,
    const T* __restrict__ k_in,
    const T* __restrict__ v_in,
    const float* __restrict__ cos_sin_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    const int32_t* __restrict__ seq_idx_map,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq
) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens) return;

    int seq_idx = seq_idx_map ? seq_idx_map[token_idx] : token_idx;
    int position = context_lens ? (context_lens[seq_idx] - 1) : token_idx;

    const float* cos_row = cos_sin_cache + position * head_dim;
    const float* sin_row = cos_sin_cache + position * head_dim + (head_dim / 2);

    int half_rot = head_dim / 2;

    // 1. Process Query Heads
    if (head_idx < num_heads) {
        const T* q_src = q_in + (token_idx * num_heads + head_idx) * head_dim;
        T* q_dst = q_out + (token_idx * num_heads + head_idx) * head_dim;

        for (int i = tid; i < half_rot; i += blockDim.x) {
            float x0 = cuda::to_float(q_src[i]);
            float x1 = cuda::to_float(q_src[i + half_rot]);
            float cos_val = cos_row[i];
            float sin_val = sin_row[i];

            float r0 = x0 * cos_val - x1 * sin_val;
            float r1 = x0 * sin_val + x1 * cos_val;

            q_dst[i] = cuda::from_float<T>(r0);
            q_dst[i + half_rot] = cuda::from_float<T>(r1);
        }
    }

    // 2. Process KV Heads & Write directly to Paged Cache
    if (head_idx < num_kv_heads && block_tables != nullptr) {
        int block_num = position / block_size;
        int block_offset = position % block_size;
        int physical_block_id = block_tables[seq_idx * max_blocks_per_seq + block_num];

        const T* k_src = k_in + (token_idx * num_kv_heads + head_idx) * head_dim;
        const T* v_src = v_in + (token_idx * num_kv_heads + head_idx) * head_dim;

        // Strides in physical KV cache: [num_blocks, num_kv_heads, block_size, head_dim]
        int kv_cache_offset = ((physical_block_id * num_kv_heads + head_idx) * block_size + block_offset) * head_dim;
        T* k_dst = k_cache + kv_cache_offset;
        T* v_dst = v_cache + kv_cache_offset;

        for (int i = tid; i < half_rot; i += blockDim.x) {
            float k0 = cuda::to_float(k_src[i]);
            float k1 = cuda::to_float(k_src[i + half_rot]);
            float cos_val = cos_row[i];
            float sin_val = sin_row[i];

            float rk0 = k0 * cos_val - k1 * sin_val;
            float rk1 = k0 * sin_val + k1 * cos_val;

            k_dst[i] = cuda::from_float<T>(rk0);
            k_dst[i + half_rot] = cuda::from_float<T>(rk1);

            // Value does not undergo RoPE rotation
            v_dst[i] = v_src[i];
            v_dst[i + half_rot] = v_src[i + half_rot];
        }
    }
}

void launch_fused_rope_paged(const FusedRoPEParams& params, cudaStream_t stream) {
    dim3 grid(params.num_tokens, max(params.num_heads, params.num_kv_heads));
    dim3 block(min(128, params.head_dim / 2));

    if (params.dtype == CUDA_R_16F) {
        fused_rope_paged_kernel<half><<<grid, block, 0, stream>>>(
            static_cast<half*>(params.q_out),
            static_cast<half*>(params.k_cache),
            static_cast<half*>(params.v_cache),
            static_cast<const half*>(params.q_in),
            static_cast<const half*>(params.k_in),
            static_cast<const half*>(params.v_in),
            params.cos_sin_cache,
            params.block_tables,
            params.context_lens,
            params.seq_idx_map,
            params.num_tokens,
            params.num_heads,
            params.num_kv_heads,
            params.head_dim,
            params.block_size,
            params.max_blocks_per_seq
        );
    } else {
        fused_rope_paged_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<float*>(params.q_out),
            static_cast<float*>(params.k_cache),
            static_cast<float*>(params.v_cache),
            static_cast<const float*>(params.q_in),
            static_cast<const float*>(params.k_in),
            static_cast<const float*>(params.v_in),
            params.cos_sin_cache,
            params.block_tables,
            params.context_lens,
            params.seq_idx_map,
            params.num_tokens,
            params.num_heads,
            params.num_kv_heads,
            params.head_dim,
            params.block_size,
            params.max_blocks_per_seq
        );
    }
    CUDA_CHECK_LAST_ERROR();
}

} // namespace kernels
} // namespace lyapunov
