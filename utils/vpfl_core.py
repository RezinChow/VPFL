"""
Core VPFL aggregation utilities.

Implements the distribution-aware adaptive weighting used by the VPFL server:

    alpha_k = n_k^gamma * exp(beta * Sim_k) / sum_j(n_j^gamma * exp(beta * Sim_j))

where Sim_k is the cosine similarity between client k's prior and posterior
models, n_k is the client's local data size, beta is a temperature parameter,
and gamma controls the influence of data quantity.
"""

import numpy as np
from typing import List


class AdaptiveWeightCalculator:
    """
    Calculate adaptive aggregation weights based on distribution similarity
    and client data size.
    """

    def __init__(self, beta: float = 2.0, gamma: float = 0.5):
        """
        Args:
            beta: Temperature parameter (controls weight distribution sharpness)
            gamma: Data quantity weight exponent
        """
        self.beta = beta
        self.gamma = gamma

    def compute_similarity_weights(self,
                                   similarities: List[float],
                                   data_sizes: List[int]) -> List[float]:
        """
        Compute adaptive weights from similarities and data sizes.

        alpha_k = n_k^gamma * exp(beta * Sim_k) / sum_j(n_j^gamma * exp(beta * Sim_j))

        Args:
            similarities: List of cosine similarities
            data_sizes: List of client data sizes

        Returns:
            List of normalized weights
        """
        n_clients = len(similarities)

        raw_weights = []
        for sim, n_k in zip(similarities, data_sizes):
            sim_weight = np.exp(self.beta * sim)
            data_factor = (n_k ** self.gamma)
            raw_weights.append(sim_weight * data_factor)

        total = sum(raw_weights)
        if total > 0:
            normalized_weights = [w / total for w in raw_weights]
        else:
            normalized_weights = [1.0 / n_clients] * n_clients

        return normalized_weights

    def compute_distance_weights(self,
                                 distances: List[float],
                                 data_sizes: List[int]) -> List[float]:
        """
        Compute weights from distances (inverse of similarity) using
        exponential decay: sim = exp(-distance).
        """
        similarities = [np.exp(-d) for d in distances]
        return self.compute_similarity_weights(similarities, data_sizes)
