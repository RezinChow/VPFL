#!/usr/bin/env python
"""
VPFL - Main Entry Point

Usage:
    # Generate datasets first
    cd dataset && python generate_Cifar10.py && python generate_FashionMNIST.py && cd ..
    
    # Run VPFL training
    python main.py --dataset Cifar10_5_pat --global_rounds 100
    python main.py --dataset FashionMNIST_5_pat --lambda_param 10.0 --momentum 0.9
"""

import argparse
import torch
import random
import numpy as np
import os
import sys

# Add system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from system.flcore.servers.servervpfl import VPFLServer
from system.flcore.trainmodel.models import create_model


def set_seed(seed=0):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='VPFL Training')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='Cifar10_5_pat',
                        choices=['Cifar10_5_pat', 'Cifar10_10_dir',
                                 'FashionMNIST_5_pat', 'FashionMNIST_10_dir'],
                        help='Dataset name')
    parser.add_argument('--num_clients', type=int, default=5,
                        help='Number of clients (auto-detected from dataset name)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes')
    
    # VPFL hyperparameters
    parser.add_argument('--lambda_param', type=float, default=10.0,
                        help='PPD constraint strength (default: 10.0)')
    parser.add_argument('--mu', type=float, default=3.0,
                        help='Perturbation layer control (default: 3.0)')
    parser.add_argument('--beta', type=float, default=2.0,
                        help='Temperature for adaptive aggregation weighting (default: 2.0)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Data quantity exponent for adaptive weighting (default: 0.5)')
    
    # Training hyperparameters
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum - CRITICAL! (default: 0.9)')
    parser.add_argument('--local_learning_rate', type=float, default=0.005,
                        help='Learning rate (default: 0.005)')
    parser.add_argument('--lr_decay', type=float, default=1.0,
                        help='Per-round learning rate decay factor, 1.0 = no decay (default: 1.0)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size (default: 10)')
    parser.add_argument('--local_epochs', type=int, default=2,
                        help='Local epochs per round (default: 2)')
    parser.add_argument('--global_rounds', type=int, default=100,
                        help='Total global rounds (default: 100)')
    
    # System configuration
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--join_ratio', type=float, default=1.0,
                        help='Ratio of clients to join each round (default: 1.0)')
    parser.add_argument('--eval_gap', type=int, default=5,
                        help='Evaluate every N rounds (default: 5)')
    
    # Saving
    parser.add_argument('--save_folder_name', type=str, default='results',
                        help='Folder to save results (default: results)')
    
    return parser.parse_args()


def check_dataset_exists(dataset_name):
    """Check if dataset exists"""
    from utils.data_utils import check_dataset_exists
    return check_dataset_exists(dataset_name)


def infer_num_clients(dataset_name):
    """Infer number of clients from dataset name"""
    if '5_' in dataset_name:
        return 5
    elif '10_' in dataset_name:
        return 10
    return 5


def main():
    """Main training function"""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Auto-detect num_clients from dataset name
    args.num_clients = infer_num_clients(args.dataset)
    
    # Set algorithm name
    args.algorithm = 'VPFL'
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    args.device = torch.device(args.device)
    
    # Check dataset exists
    if not check_dataset_exists(args.dataset):
        print(f"\nError: Dataset '{args.dataset}' not found!")
        print(f"\nPlease generate the dataset first:")
        print(f"  cd dataset && python generate_Cifar10.py && python generate_FashionMNIST.py")
        print(f"  Or: bash dataset/generate_all_datasets.sh")
        return
    
    # Print configuration
    print("\n" + "="*70)
    print("VPFL Training Configuration")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Device: {args.device}")
    print(f"Random seed: {args.seed}")
    print()
    print("VPFL Hyperparameters:")
    print(f"  lambda_param: {args.lambda_param}")
    print(f"  mu: {args.mu}")
    print(f"  beta: {args.beta}")
    print(f"  gamma: {args.gamma}")
    print()
    print("Training Hyperparameters:")
    print(f"  momentum: {args.momentum} (CRITICAL!)")
    print(f"  local_learning_rate: {args.local_learning_rate}")
    print(f"  lr_decay: {args.lr_decay}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  local_epochs: {args.local_epochs}")
    print(f"  global_rounds: {args.global_rounds}")
    print("="*70)
    print()
    
    # Create model
    print("Creating model...")
    args.model = create_model(args.dataset, args.device)
    
    # Create server
    print("Creating server...")
    server = VPFLServer(args, times=0)
    
    # Train
    print("\nStarting training...")
    server.train()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
