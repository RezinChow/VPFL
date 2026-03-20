#!/bin/bash
# Generate all datasets for VPFL

echo "=========================================="
echo "Generating VPFL Datasets"
echo "=========================================="

# CIFAR-10
echo ""
echo "[1/2] Generating CIFAR-10 datasets..."
python generate_Cifar10.py

# Fashion-MNIST
echo ""
echo "[2/2] Generating Fashion-MNIST datasets..."
python generate_FashionMNIST.py

echo ""
echo "=========================================="
echo "All datasets generated!"
echo "=========================================="
echo ""
echo "Available datasets:"
echo "  - dataset/Cifar10_5_pat/"
echo "  - dataset/Cifar10_10_dir/"
echo "  - dataset/FashionMNIST_5_pat/"
echo "  - dataset/FashionMNIST_10_dir/"
