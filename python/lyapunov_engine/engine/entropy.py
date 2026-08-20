import math
from typing import List, Tuple, Dict
import numpy as np


def simple_string_similarity(a: str, b: str) -> float:
    """Compute normalized token overlap similarity (Jaccard similarity) as lightweight fallback."""
    set_a = set(a.lower().strip().split())
    set_b = set(b.lower().strip().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection) / float(union)


def compute_semantic_entropy(
    candidate_texts: List[str],
    similarity_threshold: float = 0.65
) -> Tuple[float, str, List[int]]:
    """Compute Semantic Entropy across candidate generations.
    
    Args:
        candidate_texts: List of N stochastic response strings.
        similarity_threshold: Overlap threshold to cluster texts into the same semantic attractor.
        
    Returns:
        entropy: Discrete Shannon entropy across semantic clusters.
        confidence_rating: "high", "moderate", or "low".
        cluster_assignments: Cluster index assigned to each candidate.
    """
    if not candidate_texts:
        return 0.0, "high", []

    n = len(candidate_texts)
    clusters: List[List[int]] = []
    cluster_assignments = [-1] * n

    # Group candidate responses into semantic clusters
    for i in range(n):
        assigned = False
        text_i = candidate_texts[i]
        for c_idx, cluster in enumerate(clusters):
            # Compare with the exemplar of the cluster (first member)
            exemplar = candidate_texts[cluster[0]]
            sim = simple_string_similarity(text_i, exemplar)
            if sim >= similarity_threshold:
                cluster.append(i)
                cluster_assignments[i] = c_idx
                assigned = True
                break
        if not assigned:
            new_c_idx = len(clusters)
            clusters.append([i])
            cluster_assignments[i] = new_c_idx

    # Compute discrete probability distribution over clusters
    num_clusters = len(clusters)
    probs = [len(c) / float(n) for c in clusters]

    # Calculate discrete Shannon entropy
    entropy = 0.0
    for p in probs:
        if p > 0.0:
            entropy -= p * math.log2(p)

    # Determine confidence rating
    # Max possible entropy is log2(n)
    max_entropy = math.log2(n) if n > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    if normalized_entropy < 0.25:
        confidence_rating = "high"
    elif normalized_entropy < 0.65:
        confidence_rating = "moderate"
    else:
        confidence_rating = "low"

    return float(entropy), confidence_rating, cluster_assignments
