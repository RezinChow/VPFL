"""
Base server for VPFL
"""

import torch
import os
import numpy as np
import copy
import time
import random
from utils.data_utils import read_client_data


class Server(object):
    """
    Base server for federated learning
    """
    
    def __init__(self, args, times):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        
        self.clients = []
        self.selected_clients = []
        
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        
        self.rs_test_acc = []
        self.rs_test_loss = []
        self.rs_train_loss = []
        
        self.times = times
        self.eval_gap = args.eval_gap
        
        self.save_folder_name = args.save_folder_name if hasattr(args, 'save_folder_name') else 'results'
        
        # Create save folder
        os.makedirs(self.save_folder_name, exist_ok=True)
    
    def set_clients(self, clientObj):
        """Initialize clients"""
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            
            client = clientObj(
                self.args,
                id=i,
                train_samples=len(train_data),
                test_samples=len(test_data)
            )
            self.clients.append(client)
    
    def select_clients(self):
        """Randomly select clients for this round"""
        selected_clients = list(np.random.choice(
            self.clients, 
            self.num_join_clients, 
            replace=False
        ))
        return selected_clients
    
    def send_models(self):
        """Send global model to all clients"""
        for client in self.clients:
            client.set_parameters(self.global_model)
    
    def receive_models(self):
        """Receive models from selected clients"""
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        
        # Normalize weights
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
    
    def aggregate_parameters(self):
        """FedAvg aggregation"""
        assert len(self.uploaded_models) > 0
        
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
    
    def add_parameters(self, w, client_model):
        """Add weighted client parameters to global model"""
        for server_param, client_param in zip(
            self.global_model.parameters(), 
            client_model.parameters()
        ):
            server_param.data += client_param.data.clone() * w
    
    def test_metrics(self):
        """Test global model"""
        num_samples = []
        tot_correct = []
        tot_loss = []
        
        for c in self.clients:
            ct, ns, cl = c.test_metrics()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            tot_loss.append(cl * 1.0)
        
        ids = [c.id for c in self.clients]
        
        return ids, num_samples, tot_correct, tot_loss
    
    def evaluate(self):
        """Evaluate global model"""
        stats = self.test_metrics()
        
        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_loss = sum(stats[3]) * 1.0 / sum(stats[1])
        
        self.rs_test_acc.append(test_acc)
        self.rs_test_loss.append(test_loss)
        
        print(f"Global Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
        
        return test_acc
    
    def save_results(self):
        """Save training results"""
        from utils.result_utils import save_results
        
        results = {
            'test_acc': self.rs_test_acc,
            'test_loss': self.rs_test_loss,
            'train_loss': self.rs_train_loss,
        }
        
        save_results(self.args, results, self.save_folder_name)
    
    def save_global_model(self):
        """Save global model"""
        model_path = os.path.join(
            self.save_folder_name,
            f"{self.dataset}_{self.algorithm}_global_model.pt"
        )
        torch.save(self.global_model.state_dict(), model_path)
        print(f"Global model saved to {model_path}")
    
    def load_model(self):
        """Load global model"""
        model_path = os.path.join(
            self.save_folder_name,
            f"{self.dataset}_{self.algorithm}_global_model.pt"
        )
        
        if os.path.exists(model_path):
            self.global_model.load_state_dict(torch.load(model_path))
            print(f"Global model loaded from {model_path}")
            return True
        else:
            print(f"No model found at {model_path}")
            return False
