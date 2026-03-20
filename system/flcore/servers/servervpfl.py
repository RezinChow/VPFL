"""
VPFL server implementation
"""

import time
from system.flcore.servers.serverbase import Server
from system.flcore.clients.clientvpfl import ClientVPFL


class VPFLServer(Server):
    """
    VPFL (Variational Perturbation Personalized Federated Learning) Server
    """
    
    def __init__(self, args, times):
        super().__init__(args, times)
        
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
            self.send_models()
            
            if i % self.eval_gap == 0:
                print(f"\n{'='*50}")
                print(f"Round number: {i}/{self.global_rounds}")
                print(f"{'='*50}")
                print("Evaluating global model...")
                self.evaluate()
            
            # Local training
            for client in self.selected_clients:
                client.train(global_round=i)
            
            # Aggregation
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
        
        # Save results and model
        self.save_results()
        self.save_global_model()
