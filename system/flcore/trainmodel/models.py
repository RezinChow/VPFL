"""
Neural network models for VPFL
"""

import torch
import torch.nn as nn


class FedAvgCNN(nn.Module):
    """
    FedAvg CNN model
    
    Supports:
    - CIFAR-10: 3 channels, 1600 dim
    - Fashion-MNIST: 1 channel, 1024 dim
    """
    
    def __init__(self, in_features=3, num_classes=10, dim=1600):
        """
        Args:
            in_features: Number of input channels (1 for MNIST, 3 for CIFAR)
            num_classes: Number of output classes
            dim: Dimension after conv layers (1600 for CIFAR, 1024 for MNIST)
        """
        super(FedAvgCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


def create_model(dataset_name, device='cuda'):
    """
    Create appropriate model for dataset
    
    Args:
        dataset_name: 'Cifar10' or 'FashionMNIST'
        device: Device to put model on
    
    Returns:
        model: Initialized model on specified device
    """
    if 'Cifar10' in dataset_name:
        model = FedAvgCNN(in_features=3, num_classes=10, dim=1600)
    elif 'FashionMNIST' in dataset_name:
        model = FedAvgCNN(in_features=1, num_classes=10, dim=1024)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return model.to(device)
