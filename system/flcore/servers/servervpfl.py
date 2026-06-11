"""
VPFL server implementation.

Implements server-side VPFL with:
- Distribution-aware adaptive weight aggregation
  (cosine similarity between prior/posterior + data quantity)
- Server-side variational perturbation after aggregation
- Optional per-round learning rate decay for client optimizers
"""

import copy
import time
import torch
from system.flcore.servers.serverbase import Server
from system.flcore.clients.clientvpfl import ClientVPFL
from utils.ppd import PPDCalculator
from utils.perturbation import VariationalPerturbation
from utils.vpfl_core import AdaptiveWeightCalculator


class VPFLServer(Server):
    """
    VPFL (Variational Perturbation Personalized Federated Learning) Server.

    VPFL round:
    1. Select clients and send the global model (prior)
    2. Clients train locally and apply PPD-guided updates
    3. Receive posterior models (and their priors)
    4. Aggregate with distribution-aware adaptive weights
    5. Apply variational perturbation to the aggregated global model
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        # VPFL-specific hyperparameters
        self.lambda_param = args.lambda_param
        self.mu = args.mu
        self.beta = getattr(args, 'beta', 2.0)
        self.gamma = getattr(args, 'gamma', 0.5)
        self.lr_decay = getattr(args, 'lr_decay', 1.0)
        self.base_lr = args.local_learning_rate

        # VPFL server modules
        self.ppd_calculator = PPDCalculator(device=self.device)
        self.perturbation = VariationalPerturbation(device=self.device, mu=self.mu)
        self.weight_calculator = AdaptiveWeightCalculator(
            beta=self.beta,
            gamma=self.gamma
        )

        # Prior models corresponding to uploaded posterior models
        self.prior_models = []

        # Set up clients
        self.set_clients(ClientVPFL)

        print(f"\nJoin ratio / total clients: {self.num_join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")

        # Load model if exists
        self.load_model()

        self.Budget = []

    def train(self):
        """Main training loop"""
        for i in range(self.global_rounds + 1):
            s_t = time.time()

            self.selected_clients = self.select_clients()

            # Apply per-round learning rate decay to client optimizers
            if self.lr_decay < 1.0:
                current_lr = self.base_lr * (self.lr_decay ** i)
                for client in self.selected_clients:
                    for param_group in client.optimizer.param_groups:
                        param_group['lr'] = current_lr

            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n{'='*50}")
                print(f"Round number: {i}/{self.global_rounds}")
                print(f"{'='*50}")
                print("Evaluating global model...")
                self.evaluate()

            # Local training (includes PPD-guided update)
            for client in self.selected_clients:
                client.train(global_round=i)

            # Aggregation with adaptive weights + perturbation
            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print(f"Time cost: {self.Budget[-1]:.2f}s")

        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)

        # Final evaluation
        print("\nFinal evaluation:")
        self.evaluate()

        # Print statistics
        if len(self.rs_test_acc) > 0:
            print(f"\nBest accuracy: {max(self.rs_test_acc):.4f}")
            print(f"Final accuracy: {self.rs_test_acc[-1]:.4f}")

        if len(self.Budget) > 1:
            print(f"\nAverage time cost per round: {sum(self.Budget[1:])/len(self.Budget[1:]):.2f}s")

        # Save results, full round history and model
        self.save_results()
        self.save_history()
        self.save_global_model()

    def receive_models(self):
        """
        Receive posterior models from selected clients, along with the prior
        models needed for PPD computation and similarity weighting.
        """
        super().receive_models()

        self.prior_models = []
        for client in self.selected_clients:
            self.prior_models.append(client.prior_model)

    def aggregate_parameters(self):
        """
        Aggregate client models with the VPFL strategy.

        1. Compute cosine similarity between each client's prior and posterior
        2. Compute adaptive weights: alpha_k ~ n_k^gamma * exp(beta * Sim_k)
        3. Weighted aggregation into the global model
        4. Apply variational perturbation based on PPD between the average
           prior and the aggregated global model
        """
        assert len(self.uploaded_models) > 0

        # Step 1: similarities for adaptive weighting
        similarities = []
        data_sizes = [c.train_samples for c in self.selected_clients]

        for prior_model, posterior_model in zip(self.prior_models, self.uploaded_models):
            sim = self.ppd_calculator.compute_cosine_similarity(
                prior_model,
                posterior_model
            )
            similarities.append(sim)

        # Step 2: adaptive weights
        weights = self.weight_calculator.compute_similarity_weights(
            similarities,
            data_sizes
        )

        # Step 3: weighted aggregation
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(weights, self.uploaded_models):
            self.add_parameters(w, client_model)

        # Step 4: variational perturbation on the aggregated global model
        avg_prior = self._average_model(self.prior_models)
        ppd_matrices = self.ppd_calculator.compute_ppd(
            avg_prior,
            self.global_model
        )
        self.perturbation.apply_perturbations(
            self.global_model,
            ppd_matrices,
            order_mode='auto'
        )

    def _average_model(self, models):
        """Simple (unweighted) average of a list of models."""
        if not models:
            raise ValueError("No models to average")

        avg_model = copy.deepcopy(models[0])
        for name, param in avg_model.named_parameters():
            param.data = torch.zeros_like(param.data)

        for model in models:
            model_state = model.state_dict()
            for name, param in avg_model.named_parameters():
                param.data += model_state[name].data.clone() / len(models)

        return avg_model
