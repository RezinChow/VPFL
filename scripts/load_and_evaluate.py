#!/usr/bin/env python
"""
Load and evaluate a saved VPFL model

Usage:
    python scripts/load_and_evaluate.py --dataset Cifar10_5_pat --model_path results/Cifar10_5_pat_VPFL_global_model.pt
"""

import argparse
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.flcore.trainmodel.models import create_model
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader
import torch.nn as nn


def evaluate_model(model, dataset, device='cuda'):
    """Evaluate model on dataset"""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / total
    
    return accuracy, avg_loss


def main():
    parser = argparse.ArgumentParser(description='Load and evaluate VPFL model')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., Cifar10_5_pat)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating model for {args.dataset}...")
    model = create_model(args.dataset, device)
    
    # Load weights
    print(f"Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded successfully!")
    
    # Evaluate on each client
    print("\nEvaluating on each client:")
    print("-" * 50)
    
    num_clients = 5 if '5_' in args.dataset else 10
    accuracies = []
    
    for client_id in range(num_clients):
        # Load test data for this client
        test_data = read_client_data(args.dataset, client_id, is_train=False)
        
        # Evaluate
        acc, loss = evaluate_model(model, test_data, device)
        accuracies.append(acc)
        
        print(f"Client {client_id}: Accuracy = {acc:.4f}, Loss = {loss:.4f}")
    
    # Print summary
    print("-" * 50)
    print(f"Mean Accuracy: {sum(accuracies)/len(accuracies):.4f}")
    print(f"Best Accuracy: {max(accuracies):.4f}")
    print(f"Worst Accuracy: {min(accuracies):.4f}")


if __name__ == "__main__":
    main()
