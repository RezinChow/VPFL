"""
VPFL client implementation
"""

import copy
import torch
import torch.nn as nn
import numpy as np
import time
from system.flcore.clients.clientbase import Client


class ClientVPFL(Client):
    """
    VPFL client with personalized training
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # VPFL specific parameters
        self.lambda_param = args.lambda_param
        self.mu = args.mu
        self.perturb_scale = args.perturb_scale
        self.warmup_rounds = args.warmup_rounds
        self.perturb_start_round = args.perturb_start_round
        self.epsilon = 1e-8
        
        # Historical model
        self.history_model = copy.deepcopy(self.model)
    
    def compute_ppd(self, global_model, history_model):
        """
        Compute Prior-Posterior Distance (PPD)
        
        Formula: Γ = W_global - W_history
        """
        global_params = list(global_model.parameters())
        history_params = list(history_model.parameters())
        
        ppd_list = []
        for g_p, h_p in zip(global_params, history_params):
            ppd = g_p.data - h_p.data
            ppd_list.append(ppd)
        
        return ppd_list
    
    def compute_update_limit(self, ppd_list, global_round):
        """
        Compute update limit c
        
        Formula: c = (1/λ) * min(max(|Γ_i|))
        """
        if global_round < self.warmup_rounds:
            return float('inf')
        
        max_per_layer = [torch.max(torch.abs(ppd)).item() for ppd in ppd_list]
        min_of_max = min(max_per_layer)
        c = min_of_max / (self.lambda_param + self.epsilon)
        return c
    
    def constrained_update(self, model, ppd_list, c):
        """
        Constrained update based on PPD
        
        Formula: local = global ⊖ clamp(PPD, -c, c)
        """
        if c == float('inf'):
            return
        
        params = list(model.parameters())
        for param, ppd in zip(params, ppd_list):
            constrained_ppd = torch.clamp(ppd, -c, c)
            param.data = param.data - constrained_ppd
    
    def apply_perturbation(self, model, ppd_list, global_round):
        """
        Variational perturbation based on PPD
        """
        if global_round < self.perturb_start_round:
            return
        
        params = list(model.parameters())
        
        for param, ppd in zip(params, ppd_list):
            abs_ppd = torch.abs(ppd)
            median_val = torch.median(abs_ppd)
            
            layer_var = torch.var(param.data).item()
            layer_var = min(layer_var, 1.0)
            
            std = np.sqrt(layer_var + self.epsilon) * self.perturb_scale
            base_noise = torch.randn_like(param.data) * std
            
            # Layered perturbation
            low_order_mask = abs_ppd > median_val
            high_order_mask = ~low_order_mask
            
            perturbation = torch.zeros_like(param.data)
            perturbation = torch.where(low_order_mask, base_noise / self.mu, perturbation)
            perturbation = torch.where(high_order_mask, base_noise, perturbation)
            
            param.data = param.data + perturbation
    
    def train(self, global_round=0):
        """Train with VPFL personalization"""
        trainloader = self.load_train_data()
        
        # Start from global model
        local_model = copy.deepcopy(self.model)
        local_model.train()
        
        # Compute PPD and apply constrained update
        ppd_list = self.compute_ppd(local_model, self.history_model)
        c = self.compute_update_limit(ppd_list, global_round)
        self.constrained_update(local_model, ppd_list, c)
        
        # Apply variational perturbation
        self.apply_perturbation(local_model, ppd_list, global_round)
        
        # Local training with momentum optimizer
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.learning_rate,
            momentum=self.args.momentum
        )
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        
        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                output = local_model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
        
        # Update local model
        self.model = local_model
        
        # Update history model
        self.history_model = copy.deepcopy(local_model)
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
