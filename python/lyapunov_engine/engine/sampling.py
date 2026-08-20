from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 128
    ignore_eos: bool = False


def sample_next_tokens(
    logits: torch.Tensor, sampling_params_list: list[SamplingParams]
) -> list[int]:
    """Sample next token IDs from unnormalized logits.

    Args:
        logits: Tensor of shape [batch_size, vocab_size]
        sampling_params_list: List of SamplingParams per sequence

    Returns:
        List of sampled token IDs
    """
    batch_size = logits.shape[0]
    next_tokens = []

    for b in range(batch_size):
        sp = sampling_params_list[b]
        b_logits = logits[b].clone()

        if sp.temperature <= 0.0 or sp.temperature < 1e-4:
            # Greedy decoding
            token_id = int(torch.argmax(b_logits).item())
            next_tokens.append(token_id)
            continue

        # Temperature scaling
        b_logits = b_logits / sp.temperature

        # Top-K filtering
        if sp.top_k > 0 and sp.top_k < b_logits.shape[-1]:
            topk_vals, _ = torch.topk(b_logits, sp.top_k)
            min_val = topk_vals[-1]
            b_logits = torch.where(
                b_logits < min_val,
                torch.tensor(-float("Inf"), device=b_logits.device),
                b_logits,
            )

        # Top-P (Nucleus) filtering
        if sp.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(b_logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > sp.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            b_logits[indices_to_remove] = -float("Inf")

        probs = torch.softmax(b_logits, dim=-1)
        token_id = int(torch.multinomial(probs, num_samples=1).item())
        next_tokens.append(token_id)

    return next_tokens
