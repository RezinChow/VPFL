"""
Result utilities for VPFL
"""

import json
import os
import numpy as np
from datetime import datetime


def save_results(args, results, save_folder='results'):
    """Save training results"""
    os.makedirs(save_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.dataset}_{args.algorithm}_{timestamp}.json"
    filepath = os.path.join(save_folder, filename)
    
    save_data = {
        'dataset': args.dataset,
        'algorithm': args.algorithm,
        'num_clients': args.num_clients,
        'global_rounds': args.global_rounds,
        'local_epochs': args.local_epochs,
        'vpfl_config': {
            'lambda_param': args.lambda_param,
            'mu': args.mu,
            'perturb_scale': args.perturb_scale,
            'warmup_rounds': args.warmup_rounds,
        },
        'results': {
            'test_acc': results['test_acc'],
            'train_loss': results['train_loss'],
            'best_acc': max(results['test_acc']) if results['test_acc'] else 0,
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"Results saved to {filepath}")


def average_data(save_folder='results'):
    """Average results from multiple runs"""
    pass  # To be implemented


def print_results(results):
    """Print results in a nice format"""
    print("\n" + "="*70)
    print("Training Results")
    print("="*70)
    
    if 'test_acc' in results and results['test_acc']:
        test_acc = results['test_acc']
        print(f"Best Accuracy: {max(test_acc):.4f}")
        print(f"Final Accuracy: {test_acc[-1]:.4f}")
    
    if 'train_loss' in results and results['train_loss']:
        train_loss = results['train_loss']
        print(f"Final Loss: {train_loss[-1]:.4f}")
    
    print("="*70)
