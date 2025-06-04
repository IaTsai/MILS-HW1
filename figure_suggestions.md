# Figure and Visualization Suggestions for Technical Report

## Essential Figures to Generate

### Figure 1: Task A - Channel Combination Performance Radar Chart
**Description**: Hexagonal radar chart showing performance across all channel modes
- **Axes**: RGB, RG, GB, RB, R, G, B
- **Values**: Accuracy percentages
- **Highlight**: RGB at 33.56% as reference
- **Purpose**: Visualize relative performance degradation

### Figure 2: Task B - Architecture Performance vs Parameters Scatter Plot
**Description**: Log-scale scatter plot of all 10 architectures
- **X-axis**: Parameters (millions, log scale)
- **Y-axis**: Validation accuracy (%)
- **Annotations**: Architecture names
- **Highlight**: Wide ResNet (57.56%) and target line (48.40%)
- **Size**: Bubble size proportional to FLOPs

### Figure 3: Training Convergence Comparison
**Description**: Multi-line plot showing validation accuracy over epochs
- **Lines**: Top 5 architectures + ResNet34 baseline
- **X-axis**: Epochs (1-10)
- **Y-axis**: Validation accuracy (%)
- **Highlight**: Wide ResNet's superior convergence
- **Include**: Shaded confidence intervals

### Figure 4: Ablation Study Heatmap
**Description**: 2D heatmap showing impact of different components
- **Rows**: Different ablation configurations
- **Columns**: Metrics (Accuracy, Parameters, FLOPs)
- **Color scale**: Performance relative to baseline
- **Annotations**: Exact values in cells

### Figure 5: Computational Efficiency Analysis
**Description**: Three-panel comparison
- **Panel A**: Accuracy vs Inference Time
- **Panel B**: Accuracy vs Memory Usage
- **Panel C**: Accuracy per Million Parameters
- **Highlight**: Pareto frontier of efficient architectures

### Figure 6: Dynamic Convolution Mechanism Diagram
**Description**: Architectural diagram showing:
- Input with variable channels (1, 2, or 3)
- Weight generation network
- Dynamic convolution operation
- Feature map output
- **Style**: Clean block diagram with arrows

### Figure 7: Wide ResNet Architecture Visualization
**Description**: Network architecture diagram
- **Blocks**: 4 layers clearly labeled
- **Annotations**: Channel dimensions and operations
- **Comparison**: Side-by-side with ResNet34 for scale

### Figure 8: Error Analysis Confusion Matrix
**Description**: 10×10 confusion matrix for top confused classes
- **Color scale**: Confusion frequency
- **Annotations**: Class names and percentages
- **Highlight**: Main diagonal and major confusion patterns

## Additional Supporting Figures

### Figure S1: Per-Class Accuracy Distribution
- Histogram of class-wise accuracies
- Highlight best/worst performing classes

### Figure S2: Gradient Flow Analysis
- Gradient magnitude across layers during training
- Compare Wide ResNet vs ResNet34

### Figure S3: Feature Map Visualizations
- Sample activations from each layer
- Show diversity of learned features

### Figure S4: Training Loss Curves
- Training and validation loss over epochs
- All architectures for completeness

## Visualization Guidelines

### Color Scheme
- **Primary**: Blue (#1f77b4) for main results
- **Secondary**: Orange (#ff7f0e) for comparisons
- **Success**: Green (#2ca02c) for targets achieved
- **Baseline**: Gray (#7f7f7f) for reference lines

### Typography
- **Title**: 14pt bold
- **Axes labels**: 12pt
- **Annotations**: 10pt
- **Legend**: 10pt

### Layout
- **Size**: 6×4 inches for single plots
- **DPI**: 300 for publication quality
- **Format**: PDF vector graphics preferred

## Code Snippets for Key Visualizations

### Radar Chart (Figure 1)
```python
import matplotlib.pyplot as plt
import numpy as np

channels = ['RGB', 'RG', 'GB', 'RB', 'R', 'G', 'B']
values = [33.56, 32.22, 32.67, 32.67, 28.67, 27.33, 26.22]
angles = np.linspace(0, 2*np.pi, len(channels), endpoint=False).tolist()

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
ax.plot(angles + [angles[0]], values + [values[0]], 'o-', linewidth=2)
ax.fill(angles + [angles[0]], values + [values[0]], alpha=0.25)
ax.set_xticks(angles)
ax.set_xticklabels(channels)
ax.set_ylim(0, 40)
ax.set_title('Channel Combination Performance', size=14, weight='bold')
ax.grid(True)
```

### Scatter Plot (Figure 2)
```python
import matplotlib.pyplot as plt
import numpy as np

# Data for architectures
architectures = ['Wide ResNet', 'Wide ConvNeXt', 'ResNeSt-4Layer', ...]
params = [18.5, 1.8, 6.5, ...]  # in millions
accuracy = [57.56, 44.67, 43.78, ...]
flops = [3.21, 0.82, 1.95, ...]  # in GFLOPs

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(params, accuracy, s=np.array(flops)*100, alpha=0.6)

# Add labels
for i, txt in enumerate(architectures):
    ax.annotate(txt, (params[i], accuracy[i]), fontsize=9)

# Add target line
ax.axhline(y=48.40, color='red', linestyle='--', label='Target (48.40%)')
ax.axhline(y=53.78, color='green', linestyle='--', label='ResNet34 (53.78%)')

ax.set_xscale('log')
ax.set_xlabel('Parameters (millions)', fontsize=12)
ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax.set_title('Architecture Efficiency Analysis', fontsize=14, weight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
```

## Data Tables for Figures

All numerical data for generating these figures is available in the technical report:
- Table 1: Task A channel performance data
- Table 4: Complete architecture comparison
- Table 6: Ablation study results
- Table 7: Efficiency metrics

These visualizations will effectively communicate the key findings and support the theoretical and empirical claims in the report.