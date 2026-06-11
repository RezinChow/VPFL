"""
Base server for VPFL
"""

import torch
import os
import json
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

        # Per-round evaluation history (dumped to JSON, see save_history)
        self.rounds_history = []

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
        """Test all client models"""
        num_samples = []
        tot_correct = []
        tot_loss = []

        for c in self.clients:
            # Client returns (test_acc, test_loss, test_num)
            ct, cl, ns = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_loss.append(cl * 1.0)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_loss

    def evaluate(self):
        """
        Evaluate client (personalized) models and the global model.

        Records, per evaluation round:
        - avg_acc / std_acc: average and std of per-client accuracy
        - worst5_acc: 5th percentile of per-client accuracy
        - pgap: personalization gap (local acc - global acc, averaged)
        - global_avg_acc: global model accuracy averaged per client
        and dumps a JSON entry under <save_folder>/<dataset>_<algo>_seed<times>/.
        """
        stats = self.test_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_loss = sum(stats[3]) * 1.0 / sum(stats[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]

        self.rs_test_acc.append(test_acc)
        self.rs_test_loss.append(test_loss)

        print(f"Averaged Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
        print(f"Std Test Accuracy: {np.std(accs):.4f}")

        # Personalization Gap (PGap) and global model per-client accuracy:
        # local model = c.model (post-training), global model = self.global_model
        try:
            pgaps = []
            global_accs_per_client = []
            self.global_model.eval()
            for c in self.clients:
                testloader = c.load_test_data()
                local_correct, local_total = 0, 0
                global_correct, global_total = 0, 0
                c.model.eval()
                with torch.no_grad():
                    for x, y in testloader:
                        x = x.to(self.device)
                        y = y.to(self.device)
                        local_correct += (c.model(x).argmax(dim=1) == y).sum().item()
                        global_correct += (self.global_model(x).argmax(dim=1) == y).sum().item()
                        local_total += y.size(0)
                        global_total += y.size(0)
                local_acc = local_correct / max(local_total, 1)
                global_acc = global_correct / max(global_total, 1)
                pgaps.append(local_acc - global_acc)
                global_accs_per_client.append(global_acc)
            pgap = float(np.mean(pgaps))
            worst5 = float(np.percentile(accs, 5))
            global_avg_acc = float(np.mean(global_accs_per_client))
        except Exception as e:
            pgap = float('nan')
            worst5 = float(np.percentile(accs, 5))
            global_avg_acc = float('nan')
            global_accs_per_client = []
            print(f"[evaluate] PGap computation skipped: {e}")

        print(f"Personalization Gap: {pgap:.4f}")
        print(f"Worst-5% Test Accuracy: {worst5:.4f}")
        print(f"Global Model Avg Acc (per-client): {global_avg_acc:.4f}")

        # Per-round history entry + JSON dump
        round_idx = len(self.rs_test_acc) - 1
        history_entry = {
            "round": round_idx,
            "avg_acc": float(test_acc),
            "std_acc": float(np.std(accs)),
            "worst5_acc": worst5,
            "pgap": pgap,
            "global_avg_acc": global_avg_acc,
            "per_client_acc": [float(a) for a in accs],
            "per_client_global_acc": [float(a) for a in global_accs_per_client],
            "test_loss": float(test_loss),
        }
        self.rounds_history.append(history_entry)

        try:
            out_dir = self._history_dir()
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"round_{round_idx:04d}.json")
            with open(out_path, "w") as f:
                json.dump(history_entry, f, indent=2)
        except Exception as e:
            print(f"[evaluate] JSON dump failed: {e}")

        return test_acc

    def _history_dir(self):
        """Directory for per-round JSON history files."""
        return os.path.join(
            self.save_folder_name,
            f"{self.dataset}_{self.algorithm}_seed{self.times}"
        )

    def save_history(self):
        """Save the full per-round evaluation history to history.json."""
        try:
            out_dir = self._history_dir()
            os.makedirs(out_dir, exist_ok=True)
            history_path = os.path.join(out_dir, "history.json")
            with open(history_path, "w") as f:
                json.dump({
                    "algorithm": self.algorithm,
                    "dataset": self.dataset,
                    "seed": self.times,
                    "num_clients": self.num_clients,
                    "global_rounds": self.global_rounds,
                    "rounds": self.rounds_history,
                }, f, indent=2)
            print(f"Saved full evaluation history to {history_path}")
        except Exception as e:
            print(f"History save failed: {e}")
    
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
