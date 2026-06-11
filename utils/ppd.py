"""
Prior-Posterior Distance (PPD) utilities for VPFL.

This module implements the core innovation of VPFL: computing the difference
matrix between the global prior model and the personalized posterior model.

PPD is used to:
- Guide constrained client-side model updates
- Determine variational perturbation intensities
- Provide similarity signals for adaptive aggregation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List


class PPDCalculator:
    """
    Calculate Prior-Posterior Distance (PPD) matrices.

    The PPD matrix captures the difference between:
    - Prior: Global model received from the server
    - Posterior: Local model after training on the client's data
    """

    def __init__(self, device='cpu'):
        self.device = device

    def compute_ppd(self, prior_model: nn.Module, posterior_model: nn.Module) -> List[torch.Tensor]:
        """
        Compute Prior-Posterior Distance for each layer.

        Formula: Gamma_i = w_posterior - w_prior

        Args:
            prior_model: Global model (prior)
            posterior_model: Local model after training (posterior)

        Returns:
            List of PPD tensors, one per layer
        """
        prior_params = list(prior_model.parameters())
        posterior_params = list(posterior_model.parameters())

        ppd_matrices = []
        for p_prior, p_posterior in zip(prior_params, posterior_params):
            ppd = p_posterior.data - p_prior.data
            ppd_matrices.append(ppd)

        return ppd_matrices

    def compute_layer_statistics(self, ppd_matrices: List[torch.Tensor]) -> Dict[str, List]:
        """
        Compute statistics for each layer's PPD.

        Used for determining perturbation intensities.
        """
        stats = {
            'max_abs': [],
            'min_abs': [],
            'median_abs': [],
            'variance': []
        }

        for ppd in ppd_matrices:
            abs_ppd = torch.abs(ppd)
            stats['max_abs'].append(torch.max(abs_ppd).item())
            stats['min_abs'].append(torch.min(abs_ppd).item())
            stats['median_abs'].append(torch.median(abs_ppd).item())
            stats['variance'].append(torch.var(ppd).item())

        return stats

    def compute_update_constraint(self, ppd_matrices: List[torch.Tensor], lamda: float = 5.0) -> float:
        """
        Compute the update constraint coefficient c.

        Formula: c = 1 / (lambda * min(max(|Gamma_i|)))

        This limits the maximum step size for PPD-guided model updates.

        Args:
            ppd_matrices: List of PPD matrices per layer
            lamda: Hyperparameter controlling update magnitude

        Returns:
            Constraint coefficient c
        """
        max_abs_values = []
        for ppd in ppd_matrices:
            abs_ppd = torch.abs(ppd)
            max_abs_values.append(torch.max(abs_ppd).item())

        min_max_abs = min(max_abs_values) if max_abs_values else 1.0
        c = 1.0 / (lamda * min_max_abs) if min_max_abs > 0 else 1.0

        return c

    def compute_cosine_similarity(self, prior_model: nn.Module, posterior_model: nn.Module) -> float:
        """
        Compute average per-layer cosine similarity between prior and posterior models.

        Used as the similarity signal for adaptive aggregation weighting.

        Returns:
            Average cosine similarity across layers
        """
        prior_params = list(prior_model.parameters())
        posterior_params = list(posterior_model.parameters())

        similarities = []
        for p_prior, p_posterior in zip(prior_params, posterior_params):
            p_flat = p_prior.data.flatten()
            pos_flat = p_posterior.data.flatten()

            cos_sim = torch.nn.functional.cosine_similarity(
                p_flat.unsqueeze(0),
                pos_flat.unsqueeze(0),
                dim=1
            )
            similarities.append(cos_sim.item())

        return float(np.mean(similarities))


class DistributionAwareness:
    """
    Distribution awareness module for VPFL.

    Estimates client data distributions and computes similarity metrics
    that can be used for adaptive aggregation weighting.
    """

    def __init__(self, feature_dim: int = 32, device: str = 'cpu'):
        self.feature_dim = feature_dim
        self.device = device
        self.global_distribution = None

    def compute_local_distribution(self,
                                   model: nn.Module,
                                   data_loader: torch.utils.data.DataLoader,
                                   num_samples: int = 100) -> torch.Tensor:
        """
        Compute a local data distribution feature vector.

        d_k = (1/n_k) * sum(phi(x_i))
        """
        model.eval()
        features = []

        with torch.no_grad():
            count = 0
            for x, y in data_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)

                # Extract intermediate features
                if hasattr(model, 'feature_extractor'):
                    feat = model.feature_extractor(x)
                else:
                    # For simple CNN models, use the first conv block output
                    feat = model.conv1(x) if hasattr(model, 'conv1') else x.flatten(start_dim=1)

                feat = feat.mean(dim=0)
                features.append(feat.cpu())

                count += x.size(0)
                if count >= num_samples:
                    break

        if features:
            dist_vector = torch.stack(features).mean(dim=0).flatten()
            # Ensure consistent dimension
            if dist_vector.numel() > self.feature_dim:
                dist_vector = dist_vector[:self.feature_dim]
            elif dist_vector.numel() < self.feature_dim:
                dist_vector = torch.nn.functional.pad(
                    dist_vector,
                    (0, self.feature_dim - dist_vector.numel())
                )
        else:
            dist_vector = torch.zeros(self.feature_dim)

        return dist_vector

    def compute_class_distribution(self,
                                   data_loader: torch.utils.data.DataLoader,
                                   num_classes: int) -> torch.Tensor:
        """Compute the normalized class histogram for a client."""
        class_counts = torch.zeros(num_classes)

        for _, y in data_loader:
            for label in y:
                class_counts[label.item()] += 1

        total = class_counts.sum()
        if total > 0:
            class_counts = class_counts / total

        return class_counts

    def update_global_distribution(self,
                                   local_distributions: List[torch.Tensor],
                                   weights: List[float]) -> torch.Tensor:
        """
        Update the global distribution as a weighted average of local distributions.

        d_global = sum((n_k / n) * d_k)
        """
        if not local_distributions:
            return torch.zeros(self.feature_dim, device=self.device)

        weights_tensor = torch.tensor(weights, device=self.device)
        weights_tensor = weights_tensor / weights_tensor.sum()

        global_dist = torch.zeros_like(local_distributions[0])
        for dist, weight in zip(local_distributions, weights_tensor):
            global_dist += dist.to(global_dist.device) * weight.to(global_dist.device)

        self.global_distribution = global_dist
        return global_dist

    def compute_similarity(self, local_dist: torch.Tensor, global_dist: torch.Tensor) -> float:
        """Compute cosine similarity between local and global distribution vectors."""
        local_flat = local_dist.flatten().to(self.device)
        global_flat = global_dist.flatten().to(self.device)

        cos_sim = torch.nn.functional.cosine_similarity(
            local_flat.unsqueeze(0),
            global_flat.unsqueeze(0),
            dim=1
        )

        return cos_sim.item()
