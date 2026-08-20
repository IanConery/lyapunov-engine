#include "kernels/speculative.cuh"
#include "utils/cuda_utils.cuh"

namespace lyapunov {
namespace kernels {

// Speculative Verification Kernel
// Runs rejection sampling for each sequence in parallel
__global__ void
speculative_verification_kernel(const float *__restrict__ target_probs,
                                const float *__restrict__ draft_probs,
                                const int32_t *__restrict__ draft_tokens,
                                const float *__restrict__ rand_uniform,
                                int32_t *__restrict__ accepted_tokens,
                                int32_t *__restrict__ num_accepted_out,
                                int batch_size, int num_draft_tokens,
                                int vocab_size) {
  int seq_idx = blockIdx.x;
  if (seq_idx >= batch_size)
    return;

  // Sequential verification across K draft tokens for this sequence
  int accepted_count = 0;

  for (int k = 0; k < num_draft_tokens; ++k) {
    int token_id = draft_tokens[seq_idx * num_draft_tokens + k];
    int prob_offset = (seq_idx * num_draft_tokens + k) * vocab_size + token_id;

    float p_target = target_probs[prob_offset];
    float q_draft = draft_probs[prob_offset];
    float r = rand_uniform[seq_idx * num_draft_tokens + k];

    // Standard speculative sampling acceptance condition:
    // accept if r <= p(x) / q(x)
    float ratio = (q_draft > 1e-8f) ? (p_target / q_draft) : 1.0f;

    if (r <= ratio) {
      accepted_tokens[seq_idx * (num_draft_tokens + 1) + accepted_count] =
          token_id;
      accepted_count++;
    } else {
      // Rejected at position k
      break;
    }
  }

  num_accepted_out[seq_idx] = accepted_count;
}

void launch_speculative_verification(
    const float *target_probs, const float *draft_probs,
    const int32_t *draft_tokens, const float *rand_uniform,
    int32_t *accepted_tokens, int32_t *num_accepted, int batch_size,
    int num_draft_tokens, int vocab_size, cudaStream_t stream) {
  dim3 grid(batch_size);
  dim3 block(1);

  speculative_verification_kernel<<<grid, block, 0, stream>>>(
      target_probs, draft_probs, draft_tokens, rand_uniform, accepted_tokens,
      num_accepted, batch_size, num_draft_tokens, vocab_size);
  CUDA_CHECK_LAST_ERROR();
}

} // namespace kernels
} // namespace lyapunov
