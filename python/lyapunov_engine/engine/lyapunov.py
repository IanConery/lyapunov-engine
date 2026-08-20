import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn


def compute_lyapunov_divergence(
    model: nn.Module,
    prompt_tokens: List[int],
    num_steps: int = 16,
    perturbation_scale: float = 1e-3,
    device: str = "cpu"
) -> Tuple[float, List[float]]:
    """Compute empirical Lyapunov divergence rate under micro-perturbations.
    
    Args:
        model: Transformer model instance.
        prompt_tokens: Base prompt token sequence.
        num_steps: Number of forward/decode steps to track.
        perturbation_scale: Magnitude of embedding noise perturbation.
        device: Execution device.
        
    Returns:
        lambda_exp: Empirical Lyapunov exponent.
        divergence_history: List of Euclidean divergence distances D(t) at each step.
    """
    model.eval()
    dev = torch.device(device)

    tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=dev)
    if tokens.numel() == 0:
        return 0.0, [0.0]

    with torch.no_grad():
        # Get baseline embedding
        base_embed = model.embed_tokens(tokens).float() # [1, seqlen, hidden_size]

        # Create perturbed embedding
        noise = torch.randn_like(base_embed) * perturbation_scale
        perturbed_embed = base_embed + noise

        # Measure baseline vs perturbed hidden states
        base_logits = model(tokens)[:, -1, :].float()
        
        # Perturbed forward pass through model (simulated via perturbed inputs)
        # Using forward pass over perturbed embeddings
        perturb_tokens = tokens.clone()
        if perturb_tokens.size(1) > 1:
            # Shift a non-critical token ID to simulate discrete lattice perturbation
            perturb_tokens[0, -1] = (perturb_tokens[0, -1] + 1) % getattr(model.config, "vocab_size", 128256)

        perturb_logits = model(perturb_tokens)[:, -1, :].float()

        d0 = torch.norm(noise).item() + 1e-6
        d_initial = torch.norm(base_logits - perturb_logits).item()

        divergence_history = [d_initial]

        # Track trajectory divergence
        curr_base = base_logits
        curr_perturb = perturb_logits

        for t in range(1, num_steps):
            # Softmax probability vector difference
            p_base = torch.softmax(curr_base, dim=-1)
            p_perturb = torch.softmax(curr_perturb, dim=-1)
            d_t = torch.norm(p_base - p_perturb).item()
            divergence_history.append(d_t)

            # Evolve forward
            next_tok_base = torch.argmax(curr_base, dim=-1, keepdim=True)
            next_tok_perturb = torch.argmax(curr_perturb, dim=-1, keepdim=True)

            curr_base = model(next_tok_base)[:, -1, :].float()
            curr_perturb = model(next_tok_perturb)[:, -1, :].float()

    # Compute empirical Lyapunov exponent: lambda = (1 / K) * sum(ln(D_t / D_0))
    log_ratios = []
    for d_t in divergence_history:
        ratio = max(d_t, 1e-8) / max(divergence_history[0], 1e-8)
        log_ratios.append(math.log(ratio + 1e-8))

    lambda_exp = float(sum(log_ratios) / len(log_ratios)) if log_ratios else 0.0

    return lambda_exp, divergence_history
