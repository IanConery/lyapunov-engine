from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from lyapunov_engine.engine.sampling import SamplingParams, sample_next_tokens


class SpeculativeDecoder:
    """Coordinator for Speculative Decoding with parallel target verification."""

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: Optional[nn.Module] = None,
        num_draft_tokens: int = 4,
        temperature: float = 0.7
    ):
        self.target_model = target_model.eval()
        self.draft_model = draft_model.eval() if draft_model is not None else None
        self.num_draft_tokens = num_draft_tokens
        self.temperature = temperature
        self.total_drafted = 0
        self.total_accepted = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_drafted == 0:
            return 0.0
        return float(self.total_accepted) / float(self.total_drafted)

    def verify_draft_tokens(
        self,
        prefix_tokens: List[int],
        draft_tokens: List[int],
        device: str = "cpu"
    ) -> Tuple[List[int], Optional[int]]:
        """Verify draft tokens in a single target model forward pass.
        
        Args:
            prefix_tokens: Conversation / prompt history.
            draft_tokens: Sequence of K candidate draft tokens.
            device: Compute device.
            
        Returns:
            accepted_tokens: List of verified accepted tokens.
            bonus_token: Next token sampled from target distribution.
        """
        k = len(draft_tokens)
        if k == 0:
            return [], None

        dev = torch.device(device)
        full_seq = prefix_tokens + draft_tokens
        input_tensor = torch.tensor([full_seq], dtype=torch.long, device=dev)

        with torch.no_grad():
            # Single forward pass over prefix + draft tokens
            target_logits = self.target_model(input_tensor)[0] # [seq_len, vocab_size]

        # Extract target logits corresponding to draft positions
        # Target logit at (len(prefix) - 1 + i) predicts token draft_tokens[i]
        start_idx = len(prefix_tokens) - 1
        relevant_logits = target_logits[start_idx : start_idx + k] # [K, vocab_size]
        target_probs = F.softmax(relevant_logits / max(self.temperature, 1e-4), dim=-1)

        accepted_tokens = []
        rejected = False
        rejection_idx = k

        for i in range(k):
            token_id = draft_tokens[i]
            p_target = target_probs[i, token_id].item()

            # Speculative acceptance: standard greedy/temperature criterion
            # (Assuming draft prob ~ 1.0 for top-1 greedy draft tokens)
            r = torch.rand(1).item()
            if r <= p_target:
                accepted_tokens.append(token_id)
            else:
                rejected = True
                rejection_idx = i
                break

        # Update stats
        self.total_drafted += k
        self.total_accepted += len(accepted_tokens)

        # Sample bonus token
        if rejected:
            # Sample bonus token from the rejected position's target distribution
            bonus_logits = relevant_logits[rejection_idx : rejection_idx + 1]
            bonus_token = torch.argmax(bonus_logits, dim=-1).item()
        else:
            # All K tokens accepted, sample bonus token from step K+1
            if len(target_logits) >= len(full_seq):
                bonus_logits = target_logits[-1:]
                bonus_token = torch.argmax(bonus_logits, dim=-1).item()
            else:
                bonus_token = None

        return accepted_tokens, bonus_token
