#pragma once

#include "utils/cuda_utils.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace lyapunov {
namespace kernels {

// Speculative verification result per sequence
struct SpeculativeVerifyResult {
  int num_accepted;    // Number of accepted draft tokens (0 to K)
  int32_t bonus_token; // Next token sampled from target distribution after
                       // rejection or last accept
};

// Custom CUDA kernel for speculative token acceptance check
void launch_speculative_verification(
    const float *target_probs,   // [batch_size, K, vocab_size]
    const float *draft_probs,    // [batch_size, K, vocab_size]
    const int32_t *draft_tokens, // [batch_size, K]
    const float *rand_uniform,   // [batch_size, K]
    int32_t *accepted_tokens,    // [batch_size, K + 1]
    int32_t *num_accepted,       // [batch_size]
    int batch_size, int num_draft_tokens, int vocab_size,
    cudaStream_t stream = nullptr);

} // namespace kernels
} // namespace lyapunov
