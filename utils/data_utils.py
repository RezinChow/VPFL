"""
Data utilities for VPFL
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ClientDataset(Dataset):
    """Dataset for a single client"""
    
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def read_client_data(dataset_name, client_id, is_train=True):
    """
    Read data for a specific client
    
    Args:
        dataset_name: name of dataset (e.g., 'Cifar10_5_pat')
        client_id: client ID
        is_train: whether to read training data
    
    Returns:
        dataset: ClientDataset
    """
    data_split = 'train' if is_train else 'test'
    file_path = f"dataset/{dataset_name}/{data_split}/client_{client_id}.npz"
    
    data = np.load(file_path)
    images = data['images']
    labels = data['labels']
    
    return ClientDataset(images, labels)


def get_dataloader(dataset, batch_size=10, shuffle=True):
    """Get dataloader from dataset"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def check_dataset_exists(dataset_name):
    """Check if dataset exists"""
    import os
    train_path = f"dataset/{dataset_name}/train/client_0.npz"
    return os.path.exists(train_path)
