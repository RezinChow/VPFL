# VPFL: Variational Perturbation Personalized Federated Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Official implementation of **"Variational Perturbation Personalized Federated Learning via Prior-Posterior Distance"**.

VPFL is a personalized federated learning algorithm that leverages Prior-Posterior Distance (PPD) to achieve better personalization in non-IID environments.

## 🎯 Key Features

- **Three Core Components:**
  1. **PPD (Prior-Posterior Distance):** Measures the discrepancy between global and local historical models
  2. **Constrained Update:** Guides model updates based on PPD
  3. **Variational Perturbation:** Adds Gaussian noise based on PPD for robustness

- **Optimized Performance:**
  - Fashion-MNIST Pathological: **81.49%** (target: 80%) ✅
  - CIFAR-10 Dirichlet: **67.08%** (target: 65%) ✅
  - CIFAR-10 Pathological: **58.02%** (vs 51.19% baseline)

- **PFLlib-Compatible Architecture:** Easy to integrate with existing FL frameworks

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/VPFL.git
cd VPFL

# Install dependencies
pip install -r requirements.txt
```

### Generate Datasets

```bash
# Generate CIFAR-10 datasets
cd dataset
python generate_Cifar10.py

# Generate Fashion-MNIST datasets
python generate_FashionMNIST.py

# Or generate all at once
bash generate_all_datasets.sh
cd ..
```

### Training

```bash
# Train on Fashion-MNIST (5 clients, pathological)
python main.py --dataset FashionMNIST_5_pat --global_rounds 100

# Train on CIFAR-10 (5 clients, pathological)
python main.py --dataset Cifar10_5_pat --global_rounds 100

# Train with custom hyperparameters
python main.py \
    --dataset FashionMNIST_5_pat \
    --lambda_param 10.0 \
    --momentum 0.9 \
    --global_rounds 100
```

### Loading Pre-trained Models

```python
from system.flcore.trainmodel.models import create_model
import torch

# Create model
model = create_model('Cifar10_5_pat', device='cuda')

# Load pre-trained weights
model.load_state_dict(torch.load('results/Cifar10_5_pat_VPFL_global_model.pt'))

# Evaluate
model.eval()
```

## 📊 Supported Datasets

| Dataset | # Clients | Partition | Directory |
|---------|-----------|-----------|-----------|
| CIFAR-10 | 5 | Pathological | `dataset/Cifar10_5_pat/` |
| CIFAR-10 | 10 | Dirichlet(0.1) | `dataset/Cifar10_10_dir/` |
| Fashion-MNIST | 5 | Pathological | `dataset/FashionMNIST_5_pat/` |
| Fashion-MNIST | 10 | Dirichlet(0.1) | `dataset/FashionMNIST_10_dir/` |

## 🔧 Hyperparameters

### Recommended Settings (Optimized)

```python
VPFL_CONFIG = {
    'lambda_param': 10.0,      # PPD constraint strength
    'mu': 3.0,                 # Perturbation layer control
    'perturb_scale': 0.01,     # Perturbation magnitude
    'warmup_rounds': 20,       # Warmup before PPD activation
}

# Optimizer settings
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9  # CRITICAL for best performance!
)
```

### Parameter Descriptions

- **lambda_param (λ):** Controls the strength of PPD constraint. Higher values allow more personalization.
- **mu (μ):** Controls perturbation scaling across layers.
- **perturb_scale:** Base magnitude of Gaussian perturbation.
- **warmup_rounds:** Number of initial rounds without PPD.
- **momentum:** SGD momentum. **0.9 is crucial** for best performance (+5.52% improvement).

## 📁 Project Structure

```
VPFL/
├── README.md                    # This file
├── LICENSE                      # Apache 2.0 License
├── requirements.txt             # Python dependencies
├── main.py                      # Main entry point
│
├── dataset/                     # Dataset generation scripts
│   ├── generate_Cifar10.py
│   ├── generate_FashionMNIST.py
│   └── generate_all_datasets.sh
│
├── system/                      # Core algorithm (PFLlib-style)
│   └── flcore/
│       ├── servers/
│       │   ├── serverbase.py    # Base server
│       │   └── servervpfl.py    # VPFL server
│       ├── clients/
│       │   ├── clientbase.py    # Base client
│       │   └── clientvpfl.py    # VPFL client
│       └── trainmodel/
│           └── models.py        # Neural network models
│
├── utils/                       # Utilities
│   ├── data_utils.py            # Data loading utilities
│   └── result_utils.py          # Result saving utilities
│
└── results/                     # Results and saved models
```

## 🧪 Experiments

### Run All Benchmarks

```bash
bash scripts/run_experiments.sh
```

### Single Configuration

```bash
# Quick test (20 rounds)
python main.py --dataset FashionMNIST_5_pat --global_rounds 20

# Full training (100 rounds)
python main.py --dataset Cifar10_5_pat --global_rounds 100

# With model saving
python main.py --dataset FashionMNIST_5_pat --global_rounds 100
```

## 📈 Performance Comparison

| Method | CIFAR-10 Path | Fashion Path | CIFAR-10 Dirichlet |
|--------|--------------|--------------|-------------------|
| **VPFL (Ours)** | **58.02%** | **81.49%** | **67.08%** |
| FedAvg | ~50% | ~75% | ~65% |
| VPFL Baseline | 51.19% | - | - |

## 💡 Key Findings

1. **Momentum is Critical:** Adding momentum=0.9 improved accuracy by **+5.52%**
2. **λ=10.0 is Optimal:** Increasing from 5.0 to 10.0 added **+0.66%**
3. **Fashion-MNIST is Easier:** Achieved 81.49% vs 58.02% on CIFAR-10
4. **Three Components Work Together:** PPD + Constrained Update + Perturbation

## 🐛 Troubleshooting

### Dataset Not Found
```bash
# Generate datasets first
cd dataset
python generate_Cifar10.py
python generate_FashionMNIST.py
```

### Out of Memory
```bash
# Reduce batch size
python main.py --dataset Cifar10_5_pat --batch_size 5
```

### Slow Training
```bash
# Use fewer rounds
python main.py --dataset Cifar10_5_pat --global_rounds 50
```

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhou2025variational,
  title={Variational Perturbation Personalized Federated Learning via Prior-Posterior Distance},
  author={Zhou, Hefeng and Wang, Yuanbin and Wang, Jun and Lou, Jiong and Bao, Wugedele and Wu, Chentao and Li, Jie},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Architecture inspired by [PFLlib](https://github.com/TsingZ0/PFLlib)
- PyTorch team for the excellent deep learning framework
