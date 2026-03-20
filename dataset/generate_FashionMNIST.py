"""
Generate FashionMNIST dataset for VPFL

Usage:
    python generate_FashionMNIST.py
"""

import numpy as np
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms

random.seed(1)
np.random.seed(1)


def partition_data(dataset, num_clients, partition_type='pat', alpha=0.1):
    """
    Partition dataset for federated learning
    
    Args:
        dataset: (images, labels) tuple
        num_clients: number of clients
        partition_type: 'pat' (pathological) or 'dir' (dirichlet)
        alpha: dirichlet concentration parameter
    
    Returns:
        client_data: list of (client_images, client_labels) tuples
    """
    images, labels = dataset
    num_samples = len(images)
    num_classes = len(np.unique(labels))
    
    client_data = [([], []) for _ in range(num_clients)]
    
    if partition_type == 'pat':
        # Pathological partition: each client gets 2 classes
        shards_per_client = 2
        total_shards = num_clients * shards_per_client
        samples_per_shard = num_samples // total_shards
        
        # Sort by labels
        idxs = np.argsort(labels)
        
        # Create shards
        shard_idxs = list(range(total_shards))
        for i in range(num_clients):
            rand_shards = np.random.choice(shard_idxs, shards_per_client, replace=False)
            shard_idxs = [s for s in shard_idxs if s not in rand_shards]
            
            for shard in rand_shards:
                start = shard * samples_per_shard
                end = (shard + 1) * samples_per_shard
                client_data[i][0].extend(images[idxs[start:end]])
                client_data[i][1].extend(labels[idxs[start:end]])
    
    elif partition_type == 'dir':
        # Dirichlet partition
        idxs_per_class = [np.where(labels == i)[0] for i in range(num_classes)]
        
        for k in range(num_classes):
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * len(idxs_per_class[k]) for p in proportions])
            proportions = proportions.astype(int)
            proportions[-1] = len(idxs_per_class[k]) - np.sum(proportions[:-1])
            
            np.random.shuffle(idxs_per_class[k])
            idxs_split = np.split(idxs_per_class[k], np.cumsum(proportions[:-1]))
            
            for i in range(num_clients):
                client_data[i][0].extend(images[idxs_split[i]])
                client_data[i][1].extend(labels[idxs_split[i]])
    
    # Convert to numpy arrays
    for i in range(num_clients):
        client_data[i] = (np.array(client_data[i][0]), np.array(client_data[i][1]))
    
    return client_data


def split_train_test(client_data, test_ratio=0.2):
    """Split client data into train and test sets"""
    train_data = []
    test_data = []
    
    for images, labels in client_data:
        num_samples = len(images)
        num_test = int(num_samples * test_ratio)
        
        # Shuffle
        idxs = np.random.permutation(num_samples)
        
        test_idx = idxs[:num_test]
        train_idx = idxs[num_test:]
        
        test_data.append((images[test_idx], labels[test_idx]))
        train_data.append((images[train_idx], labels[train_idx]))
    
    return train_data, test_data


def save_data(dir_path, train_data, test_data, num_clients, num_classes):
    """Save partitioned data"""
    train_path = os.path.join(dir_path, "train")
    test_path = os.path.join(dir_path, "test")
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Save train data
    for i, (images, labels) in enumerate(train_data):
        np.savez(os.path.join(train_path, f"client_{i}.npz"), 
                 images=images, labels=labels)
    
    # Save test data
    for i, (images, labels) in enumerate(test_data):
        np.savez(os.path.join(test_path, f"client_{i}.npz"), 
                 images=images, labels=labels)
    
    print(f"Saved data for {num_clients} clients")
    print(f"  Train samples per client: {[len(d[0]) for d in train_data]}")
    print(f"  Test samples per client: {[len(d[0]) for d in test_data]}")


def generate_dataset(dir_path, num_clients, partition_type='pat', alpha=0.1):
    """Generate FashionMNIST dataset"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Check if already exists
    if os.path.exists(os.path.join(dir_path, "train", "client_0.npz")):
        print(f"Dataset already exists at {dir_path}")
        return
    
    print("Downloading FashionMNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = torchvision.datasets.FashionMNIST(
        root=os.path.join(dir_path, "rawdata"), 
        train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        root=os.path.join(dir_path, "rawdata"), 
        train=False, download=True, transform=transform
    )
    
    # Combine train and test for partitioning
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    
    for train_data in trainloader:
        train_images, train_labels = train_data[0].numpy(), train_data[1].numpy()
    for test_data in testloader:
        test_images, test_labels = test_data[0].numpy(), test_data[1].numpy()
    
    # Combine all data
    all_images = np.concatenate([train_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    
    num_classes = len(np.unique(all_labels))
    print(f"Number of classes: {num_classes}")
    print(f"Total samples: {len(all_images)}")
    
    # Partition data
    print(f"Partitioning with {partition_type} method...")
    client_data = partition_data((all_images, all_labels), num_clients, partition_type, alpha)
    
    # Split train/test
    train_data, test_data = split_train_test(client_data, test_ratio=0.2)
    
    # Save
    save_data(dir_path, train_data, test_data, num_clients, num_classes)
    print(f"Dataset generation complete!")


if __name__ == "__main__":
    # Default: 5 clients, pathological partition
    generate_dataset("FashionMNIST_5_pat/", num_clients=5, partition_type='pat')
    
    # 10 clients, dirichlet partition
    generate_dataset("FashionMNIST_10_dir/", num_clients=10, partition_type='dir', alpha=0.1)
