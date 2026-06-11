"""
VPFL client implementation.

Implements client-side VPFL with:
- Prior synchronization (client always continues from the received global model)
- Standard local SGD training (momentum + weight decay)
- Post-training Prior-Posterior Distance (PPD) guided update:
      h_i <- h_i (-) constrained(Gamma_i)
"""

import copy
import time
import torch
from system.flcore.clients.clientbase import Client
from utils.ppd import PPDCalculator


class ClientVPFL(Client):
    """
    VPFL client with PPD-guided personalized updates.

    Training procedure per round:
    1. Receive global model (prior) and sync local model to it
    2. Train locally with SGD (momentum=0.9, weight_decay=1e-4)
    3. Compute PPD: Gamma_i = posterior - prior
    4. Apply constrained PPD-guided update: param -= lr * constrained(Gamma_i)
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # VPFL specific parameters
        self.lambda_param = args.lambda_param
        self.epsilon = 1e-8

        # Recreate optimizer with momentum and weight decay
        # (momentum=0.9 + weight_decay=1e-4 are critical for convergence)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=getattr(args, 'momentum', 0.9),
            weight_decay=1e-4
        )

        # PPD calculator
        self.ppd_calculator = PPDCalculator(device=self.device)

        # Prior model (global model received from server)
        self.prior_model = None

    def set_parameters(self, model):
        """
        Receive the global model from the server.

        Stores a deep copy as the prior for PPD computation and syncs the
        local model so training continues from the global model (prior-sync
        fix: without this the client would train from a stale local state).
        """
        self.prior_model = copy.deepcopy(model)
        super().set_parameters(model)

    def train(self, global_round=0):
        """
        Perform local training followed by the PPD-guided update.
        """
        trainloader = self.load_train_data()
        self.model.train()

        start_time = time.time()

        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        # After training, apply the PPD-guided update
        self._apply_ppd_update()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _apply_ppd_update(self):
        """
        Apply the PPD-guided update to the model.

        h_i <- h_i (-) constrained(Gamma_i)

        where Gamma_i = posterior - prior, and the constraint coefficient is
        c = 1 / (lambda * min(max(|Gamma_i|))). Layers whose PPD exceeds c
        are rescaled before the subtraction step.
        """
        if self.prior_model is None:
            return

        # Compute PPD: Gamma_i = posterior - prior
        ppd_matrices = self.ppd_calculator.compute_ppd(
            self.prior_model,
            self.model
        )

        # Compute constraint coefficient
        c = self.ppd_calculator.compute_update_constraint(
            ppd_matrices, self.lambda_param
        )

        # Apply constrained update
        model_params = list(self.model.parameters())
        for param, ppd in zip(model_params, ppd_matrices):
            # Constrain step size
            abs_ppd = torch.abs(ppd)
            max_abs = torch.max(abs_ppd)
            if max_abs > c:
                ppd = ppd * (c / (max_abs + self.epsilon))

            # Update: subtract PPD ((-) operation)
            param.data = param.data - self.learning_rate * ppd

    def get_ppd(self):
        """
        Get the current Prior-Posterior Distance matrices.

        Returns:
            List of PPD tensors, or None if the prior model is not set
        """
        if self.prior_model is None:
            return None

        return self.ppd_calculator.compute_ppd(self.prior_model, self.model)
