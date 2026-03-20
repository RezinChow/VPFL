#!/bin/bash
# Run all benchmark experiments for VPFL

echo "=========================================="
echo "VPFL Benchmark Experiments"
echo "=========================================="

# Create results directory
mkdir -p results

# Configuration 1: CIFAR-10 Pathological (5 clients)
echo ""
echo "[1/4] Running CIFAR-10 Pathological..."
python main.py \
    --dataset Cifar10_5_pat \
    --lambda_param 10.0 \
    --momentum 0.9 \
    --global_rounds 100

# Configuration 2: Fashion-MNIST Pathological (5 clients)
echo ""
echo "[2/4] Running Fashion-MNIST Pathological..."
python main.py \
    --dataset FashionMNIST_5_pat \
    --lambda_param 10.0 \
    --momentum 0.9 \
    --global_rounds 100

# Configuration 3: CIFAR-10 Dirichlet (10 clients, alpha=0.1)
echo ""
echo "[3/4] Running CIFAR-10 Dirichlet..."
python main.py \
    --dataset Cifar10_10_dir \
    --lambda_param 10.0 \
    --momentum 0.9 \
    --global_rounds 100

# Configuration 4: Fashion-MNIST Dirichlet (10 clients, alpha=0.1)
echo ""
echo "[4/4] Running Fashion-MNIST Dirichlet..."
python main.py \
    --dataset FashionMNIST_10_dir \
    --lambda_param 10.0 \
    --momentum 0.9 \
    --global_rounds 100

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: results/"
echo "Models saved to: results/*_global_model.pt"
echo "=========================================="
