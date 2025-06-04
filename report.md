# MILS-HW1

## Dynamic Convolution and Efficient 4-Layer Networks for Mini-ImageNet Classification
**Author:** Ian Tsai 
**Student ID:** 313553058
**Affiliation:** NYCU 
**Date:** June 2025/06/04
Git Link: https://github.com/IaTsai/MILS-HW1
---

## Executive Summary

- **Task A**: Dynamic convolution module achieving 33.56% accuracy on RGB input
- **Task B**: Wide ResNet (4 layers) achieving **57.56% accuracy**, surpassing the 48.40% target by 19%

Key innovation: Automated architecture testing framework evaluating 10 diverse designs under identical conditions.

---

## 1. Introduction

### 1.1 Problem Statement
- **Task A**: Design a CNN handling arbitrary channel combinations (R, G, B, RG, GB, RB, RGB)
- **Task B**: Create a 4-layer network achieving ≥90% of ResNet34's performance (48.40%)

### 1.2 Challenges
- Variable input modalities requiring adaptive processing
- Severe depth constraint (4 layers vs ResNet34's 34 layers)
- Balancing model capacity with efficiency

### 1.3 Contributions
1. Novel weight generation network for dynamic channel adaptation
2. Automated testing framework for fair architecture comparison
3. Demonstration that wide shallow networks can outperform deep networks

---

## 2. Methodology

### 2.1 Task A: Dynamic Convolution Design

#### Architecture Overview
```
Input (1-3 channels) → Dynamic Conv Block → Fixed CNN Layers → Output (100 classes)
```

#### Key Components

**1. Weight Generation Network**
- Separate generators for 1, 2, and 3 channel inputs
- Dynamically produces convolution weights based on input channels
- Formula: `W = WeightGen_c(1)` where c ∈ {1, 2, 3}

**2. Channel Attention (for RGB)**
- Learns importance weights for each channel
- Applied before convolution for better feature extraction
- Helps model focus on most informative channels

**3. Unified Architecture Benefits**
- Single model handles all channel combinations
- No need for separate models or retraining
- Maintains consistent performance across modes

### 2.2 Task B: Efficient 4-Layer Networks

#### Evolution of Approach

**Phase 1: Initial Explorations**
- Mamba (5.33%) → Improved Mamba (37.33%)
- Identified need for more sophisticated architectures

**Phase 2: Lightweight Attempts**
- MiniConvNeXt (16.22%): Memory efficient but insufficient accuracy
- Realized need for systematic comparison

**Phase 3: Automated Architecture Search**
- Developed testing framework for 10 architectures
- Standardized evaluation (10 epochs each)
- Wide ResNet emerged victorious

#### Tested Architectures
1. **Wide ResNet** - Width-focused design
2. **Wide ConvNeXt** - Modern convolution blocks
3. **ResNeSt-4Layer** - Split attention mechanism
4. **Mini Swin** - Transformer approach
5. **EfficientNet-Style** - Mobile-optimized
6. **Attention-CNN** - Hybrid approach
7. **ConvMixer** - Patch-based mixing
8. **Multi-Scale CNN** - Parallel convolutions
9. **Dense-Efficient** - Dense connections
10. **Ghost-Net** - Efficient operations

---

## 3. Experimental Setup

### 3.1 Dataset
- **Mini-ImageNet**: 100 classes subset
- **Training**: 63,325 samples
- **Validation**: 12,000 samples
- **Preprocessing**: 224×224, ImageNet normalization

### 3.2 Baseline Configuration
- **Model**: ResNet34 (from scratch)
- **Achievement**: 53.78% @ 10 epochs
- **Parameters**: ~21.3M
- **Target**: 48.40% (90% of 53.78%)

### 3.3 Training Details

| Parameter | ResNet34 | Wide ResNet |
|-----------|----------|-------------|
| Optimizer | SGD (0.9) | AdamW |
| Learning Rate | 0.1 | 3e-4 |
| Weight Decay | 1e-4 | 0.05 |
| Batch Size | 256 | 24 |
| Scheduler | MultiStepLR | CosineAnnealing |
| Dropout | None | 0.3 |

---

## 4. Results

### 4.1 Task A: Channel Combination Performance

| Channel Mode | Accuracy | Relative Performance |
|--------------|----------|---------------------|
| **RGB** | **33.56%** | 100% (baseline) |
| RG | 32.22% | 96.0% |
| GB | 32.67% | 97.3% |
| RB | 32.67% | 97.3% |
| R | 28.67% | 85.4% |
| G | 27.33% | 81.4% |
| B | 26.22% | 78.1% |

**Key Findings:**
- Dual-channel modes retain >96% performance
- Red channel most informative individually
- Blue channel least informative (as expected)

### 4.2 Task B: Architecture Comparison

| Rank | Architecture | Accuracy | Parameters | vs Target |
|------|--------------|----------|------------|-----------|
| 1 | **Wide ResNet** | **57.56%** | 18.5M | **+19%**(Best!) |
| 2 | Wide ConvNeXt | 44.67% | 1.8M | -7.7% |
| 3 | ResNeSt-4Layer | 43.78% | 6.5M | -9.5% |
| 4 | Attention-CNN | 43.11% | 5.8M | -10.9% |
| 5 | EfficientNet-Style | 37.78% | 1.2M | -22.0% |

### 4.3 Efficiency Analysis

**Parameter Efficiency (Accuracy per Million Parameters):**
- Wide ResNet: 3.11
- ResNet34: 2.52
- **Improvement: 23.4%**

**Training Efficiency:**
- Wide ResNet reaches 57.56% in 10 epochs
- ResNet34 reaches 53.78% in 10 epochs
- Wide ResNet converges faster with better final performance

---

## 5. Ablation Studies

### 5.1 Wide ResNet Components

| Configuration | Accuracy | Impact |
|---------------|----------|---------|
| Full model (baseline) | 57.56% | - |
| Without dropout | 54.22% | -3.34% |
| Width = 2 (not 4) | 49.11% | -8.45% |
| SGD optimizer | 53.78% | -3.78% |
| No special init | 52.33% | -5.23% |

### 5.2 Critical Success Factors
1. **Width multiplier (4×)**: Most important factor
2. **Dropout regularization**: Prevents overfitting
3. **AdamW optimizer**: Better than SGD for this task
4. **Proper initialization**: Kaiming normal crucial

---

## 6. Discussion

### 6.1 Why Wide ResNet Succeeded

**1. Optimal Capacity Distribution**
- 4× width provides ~16× more features per layer
- Compensates for limited depth effectively

**2. Gradient Flow Advantages**
- Only 4 layers → minimal gradient degradation
- Skip connections maintain information flow

**3. Modern Training Techniques**
- AdamW handles sparse gradients better
- Cosine annealing prevents early convergence
- Dropout provides strong regularization

### 6.2 Architectural Insights

**Winners:** Traditional CNN architectures
- Wide ResNet, ConvNeXt variants perform best
- Benefit from established design principles

**Strugglers:** Modern/Complex architectures
- Transformers need more depth to shine
- Attention mechanisms underutilized in 4 layers

### 6.3 Practical Implications
- Depth isn't always necessary for good performance
- Wide shallow networks can be more parameter efficient
- Systematic testing reveals non-obvious winners

---

## 7. Limitations and Future Work

### 7.1 Current Limitations
- Limited to 10 training epochs
- Fixed 4-layer constraint
- No ensemble methods explored

### 7.2 Future Directions
1. **Extended Training**: 50-100 epochs may improve all models
2. **Architecture Search**: AutoML for optimal 4-layer design
3. **Knowledge Distillation**: Learn from deeper teachers
4. **Dynamic Depth**: Adaptive computation graphs

---

## 8. Conclusion

This project successfully demonstrates:

1. **Task A Success**: Dynamic convolution handles all channel combinations effectively
2. **Task B Excellence**: Wide ResNet achieves 57.56%, exceeding target by 19%
3. **Methodology Value**: Automated testing ensures fair, comprehensive comparison

**Key Takeaway**: Carefully designed shallow networks can outperform much deeper alternatives, challenging the "deeper is better" paradigm.

---

## References

1. Zagoruyko & Komodakis (2016). Wide Residual Networks. BMVC.
2. Liu et al. (2022). A ConvNet for the 2020s. CVPR.
3. Zhang et al. (2020). ResNeSt: Split-Attention Networks. ECCV.
4. Wu et al. (2019). Pay Less Attention with Lightweight and Dynamic Convolutions. ICLR.

---

## Appendix A: Implementation Details

### A.1 Dynamic Convolution Code Structure
```python
class DynamicConvBlock(nn.Module):
    def __init__(self, out_channels=64):
        self.weight_gen_1ch = nn.Linear(1, out_channels * 7 * 7)
        self.weight_gen_2ch = nn.Linear(2, out_channels * 2 * 7 * 7)
        self.weight_gen_3ch = nn.Linear(3, out_channels * 3 * 7 * 7)
        self.channel_attention = ChannelAttention()
```

### A.2 Wide ResNet Configuration
```python
WideResNet4Layer(
    width=4,  # 4× wider than standard
    num_classes=100,
    dropout_rate=0.3
)
```

---

## Appendix B: Reproducibility Checklist

All code available on GitHub: https://github.com/IaTsai/MILS-HW1
Random seeds fixed (42)  
Environment specifications provided  
Hyperparameters documented  
Dataset splits specified  
Evaluation metrics defined  