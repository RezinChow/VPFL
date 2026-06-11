"""
Variational Perturbation utilities for VPFL.

Implements the perturbation mechanism described in the VPFL paper:
- Low-order perturbation: N(0, Var(h_l) / mu), smaller perturbation for
  significant parameter deviation
- High-order perturbation: N(0, Var(h_l)), larger perturbation to address
  the negative impact of an increased learning rate

The perturbation follows a zero-mean distribution to avoid systematic bias.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class VariationalPerturbation:
    """
    Variational model perturbation for improving robustness.

    Applies layer-wise Gaussian perturbations based on Prior-Posterior
    Distance (PPD) statistics.
    """

    def __init__(self, device: str = 'cpu', mu: float = 3.0):
        """
        Args:
            device: 'cpu' or 'cuda'
            mu: Perturbation scale factor (default: 3.0 from paper)
        """
        self.device = device
        self.mu = mu

    def compute_layer_variance(self, ppd_matrix: torch.Tensor) -> float:
        """Compute the variance of a layer's PPD matrix, Var(h_l)."""
        return torch.var(ppd_matrix.float()).item()

    def compute_perturbation_threshold(self, ppd_matrix: torch.Tensor) -> float:
        """
        Compute the median of absolute PPD values.

        Used to determine whether to apply low- or high-order perturbation.
        """
        abs_ppd = torch.abs(ppd_matrix)
        return torch.median(abs_ppd).item()

    def generate_layer_perturbation(self,
                                    shape: Tuple[int, ...],
                                    variance: float,
                                    order: str = 'high') -> torch.Tensor:
        """
        Generate the perturbation tensor for one layer.

        Low-order:  N(0, Var(h_l) / mu)
        High-order: N(0, Var(h_l))
        """
        if order == 'low':
            scale = variance / self.mu
        else:
            scale = variance

        perturbation = torch.randn(shape, device=self.device) * np.sqrt(scale)
        return perturbation

    def determine_perturbation_order(self,
                                     ppd_matrix: torch.Tensor,
                                     threshold: Optional[float] = None) -> str:
        """
        Determine the perturbation order based on PPD magnitude.

        Low-order: when mean(|PPD|) > threshold (significant deviation)
        High-order: otherwise
        """
        if threshold is None:
            threshold = self.compute_perturbation_threshold(ppd_matrix)

        abs_ppd = torch.abs(ppd_matrix)
        mean_abs_ppd = torch.mean(abs_ppd).item()

        if mean_abs_ppd > threshold:
            return 'low'
        else:
            return 'high'

    def apply_perturbations(self,
                            model: nn.Module,
                            ppd_matrices: List[torch.Tensor],
                            order_mode: str = 'auto') -> nn.Module:
        """
        Apply perturbations to all layers of a model (in-place).

        Args:
            model: Model to perturb
            ppd_matrices: List of PPD matrices (one per layer)
            order_mode: 'auto', 'low', or 'high'

        Returns:
            Perturbed model
        """
        model_params = list(model.parameters())

        for param, ppd in zip(model_params, ppd_matrices):
            if order_mode == 'auto':
                order = self.determine_perturbation_order(ppd)
            else:
                order = order_mode

            variance = self.compute_layer_variance(ppd)
            perturbation = self.generate_layer_perturbation(
                param.shape,
                variance,
                order
            )

            param.data = param.data + perturbation.to(param.device)

        return model

    def compute_robustness_metric(self, original_loss: float, perturbed_loss: float) -> float:
        """
        Relative loss increase caused by the perturbation (lower is better).
        """
        if original_loss == 0:
            return 0.0
        return (perturbed_loss - original_loss) / original_loss
