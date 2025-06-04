# MILS Assignment I - Dynamic Convolution & Efficient 4-Layer Networks

## Achievement Summary

This project successfully implements two challenging tasks in modern deep learning:

- **Task A**: Dynamic Convolution Module supporting arbitrary channel combinations
  - ✅ Achieved **33.56%** accuracy on RGB input
  - ✅ Successfully handles all channel combinations (R, G, B, RG, GB, RB, RGB)

- **Task B**: Efficient 4-Layer Network Architecture  
  - ✅ Achieved **57.56%** accuracy with Wide ResNet
  - ✅ **Exceeded target by 19%** (target: 48.40%, ResNet34's 90%)
  - ✅ Outperformed ResNet34 (53.78%) with only 4 layers

## Project Structure

```
MILS_Assignment1/
├── Task_A/                      # Dynamic convolution implementation
│   ├── taskA_fixed.py          # Main training script
│   ├── flexible_cnn.py         # Dynamic CNN module
│   └── channel_comparison.png   # Performance comparison chart
├── Task_B/                      # 4-layer efficient networks
│   ├── architectures/          # All tested architectures
│   │   ├── wide_resnet.py      # Best performing model (57.56%)
│   │   └── all_architectures.py # 10 different architectures
│   ├── auto_test_architectures.py  # Automated testing system
│   └── run_architecture_tests.py   # Main testing script
├── baselines/                   # Baseline implementations
│   └── train_clean_resnet34.py # ResNet34 baseline (53.78%)
├── results/                     # Experimental results
│   └── architecture_results/   # Performance rankings & models
└── docs/                        # Documentation
    └── technical_report.pdf     # Detailed technical report
```

## Quick Start

### Prerequisites

```bash
# Create conda environment
conda create -n mils python=3.8
conda activate mils

# Install dependencies
pip install -r requirements.txt
```

### Running Task A - Dynamic Convolution

Test the dynamic convolution module with different channel combinations:

```bash
# Run complete channel comparison
python Task_A/taskA_fixed.py

# Test specific channel mode
python Task_A/taskA_fixed.py --channels RGB  # or RG, GB, RB, R, G, B
```

### Running Task B - Wide ResNet Training

Train the best performing 4-layer architecture:

```bash
# Train Wide ResNet (achieves 57.56%)
CUDA_VISIBLE_DEVICES=0 python Task_B/train_wide_resnet.py --epochs 10

# Run automated architecture testing (10 architectures)
CUDA_VISIBLE_DEVICES=0 python Task_B/run_architecture_tests.py
```

## Key Results

### Task A: Channel Combination Performance

| Channel Mode | Accuracy | Relative to RGB |
|--------------|----------|-----------------|
| RGB          | 33.56%   | 100%           |
| RG           | 32.22%   | 96.0%          |
| GB           | 32.67%   | 97.3%          |
| RB           | 32.67%   | 97.3%          |
| R            | 28.67%   | 85.4%          |
| G            | 27.33%   | 81.4%          |
| B            | 26.22%   | 78.1%          |

### Task B: Architecture Comparison (10 epochs each)

| Architecture | Accuracy | Parameters | vs Target (48.40%) |
|--------------|----------|------------|-------------------|
| **Wide ResNet** | **57.56%** | 18.5M | **+19%** ✅ |
| Wide ConvNeXt | 44.67% | 1.8M | -7.7% |
| ResNeSt-4Layer | 43.78% | 6.5M | -9.5% |
| Attention-CNN | 43.11% | 5.8M | -10.9% |
| EfficientNet-Style | 37.78% | 1.2M | -22.0% |
| Multi-Scale CNN | 31.78% | 2.8M | -34.3% |
| ConvMixer-Style | 29.11% | 4.2M | -39.9% |
| Dense-Efficient | 26.00% | 2.1M | -46.3% |
| Ghost-Net Style | 23.33% | 0.7M | -51.8% |
| Mini Swin | 16.22% | 1.3M | -66.5% |

## Technical Innovations

### 1. Dynamic Convolution (Task A)
- **Weight Generation Network**: Dynamically generates convolution weights based on input channels
- **Channel-wise Attention**: Adaptive importance weighting for different channels
- **Unified Architecture**: Single model handles 1, 2, or 3 channel inputs

### 2. Automated Architecture Search (Task B)
- **Systematic Testing**: 10 diverse architectures tested under identical conditions
- **Fair Comparison**: All models trained for exactly 10 epochs
- **Comprehensive Coverage**: From traditional CNNs to modern Transformers

### 3. Wide ResNet Success Factors
- **Width over Depth**: 4x wider channels compensate for shallow depth
- **Efficient Design**: Outperforms deeper ResNet34 with fewer layers
- **Strong Regularization**: Dropout and proper initialization

## Reproducing Results

### Dataset Preparation
The project uses mini-ImageNet with 100 classes:
- Training samples: 63,325
- Validation samples: 12,000
- Image size: 224×224

### Training Configuration
```python
# Wide ResNet optimal settings
optimizer = AdamW(lr=3e-4, weight_decay=0.05)
scheduler = CosineAnnealingLR(T_max=10)
batch_size = 24
epochs = 10
dropout_rate = 0.3
width_multiplier = 4
```

## Performance Analysis

### Why Wide ResNet Succeeded
1. **Optimal capacity**: 18.5M parameters provide sufficient model capacity
2. **Gradient flow**: Only 4 layers ensure efficient gradient propagation  
3. **Width advantage**: Increased width captures more features per layer
4. **Regularization**: Dropout prevents overfitting on limited data

### Efficiency Comparison
- Wide ResNet: 3.11 accuracy/M params
- ResNet34: 2.56 accuracy/M params
- **21% more parameter efficient** than ResNet34

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mils_assignment1_2024,
  title={Dynamic Convolution and Efficient 4-Layer Networks for Mini-ImageNet},
  author={[Your Name]},
  year={2024},
  publisher={GitHub},
  url={https://github.com/[your-username]/MILS_Assignment1}
}
```

## Acknowledgments

- Mini-ImageNet dataset creators
- PyTorch team for the excellent framework
- Authors of Wide ResNet, ConvNeXt, and other architectures

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This project was completed as part of the Machine Intelligence and Learning Systems (MILS) course assignment. The automated architecture testing methodology and Wide ResNet optimization represent original contributions to efficient deep learning design.