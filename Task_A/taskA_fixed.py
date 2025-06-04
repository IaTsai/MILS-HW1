"""
Fixed Task A implementation with proper channel handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms
import matplotlib.pyplot as plt


class DynamicConv2D(nn.Module):
    """Dynamic convolution that adapts to input channels"""
    def __init__(self, max_in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DynamicConv2D, self).__init__()
        self.max_in_channels = max_in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weight generator MLP
        self.weight_gen = nn.Sequential(
            nn.Linear(max_in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels * max_in_channels * kernel_size * kernel_size)
        )
        
        # Bias generator
        self.bias_gen = nn.Linear(max_in_channels, out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C <= self.max_in_channels, f"Input channels {C} exceed max limit {self.max_in_channels}!"

        # Create channel indicator
        channel_indicator = torch.zeros((B, self.max_in_channels), device=x.device)
        channel_indicator[:, :C] = 1.0

        # Generate dynamic weights
        dynamic_weights = self.weight_gen(channel_indicator)
        dynamic_weights = dynamic_weights.view(
            B, self.out_channels, self.max_in_channels, self.kernel_size, self.kernel_size
        )
        
        # Only use weights for actual input channels
        dynamic_weights = dynamic_weights[:, :, :C, :, :]
        
        # Generate dynamic bias
        dynamic_bias = self.bias_gen(channel_indicator)  # [B, out_channels]

        # Apply convolution per sample
        outputs = []
        for i in range(B):
            out = F.conv2d(
                x[i:i+1],
                dynamic_weights[i],
                bias=dynamic_bias[i],
                stride=self.stride,
                padding=self.padding
            )
            outputs.append(out)

        return torch.cat(outputs, dim=0)


class FlexibleCNN(nn.Module):
    """CNN that can handle different input channels using dynamic convolution"""
    def __init__(self, max_channels=3, num_classes=100):
        super(FlexibleCNN, self).__init__()
        
        # Dynamic first layer
        self.conv1 = DynamicConv2D(max_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Regular layers
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


class StandardCNN(nn.Module):
    """Standard CNN for comparison (requires fixed input channels)"""
    def __init__(self, in_channels, num_classes=100):
        super(StandardCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# Dataset and training utilities
CHANNEL_MAP = {
    "R": [0],
    "G": [1],
    "B": [2],
    "RG": [0, 1],
    "RB": [0, 2],
    "GB": [1, 2],
    "RGB": [0, 1, 2],
}


class MiniImageNetDataset(Dataset):
    def __init__(self, txt_path, img_root, channel_mode="RGB", transform=None):
        with open(txt_path, 'r') as f:
            self.samples = [line.strip().split() for line in f.readlines()]
        
        self.img_root = img_root
        self.channel_indices = CHANNEL_MAP[channel_mode]
        self.channel_mode = channel_mode
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Handle path format
        if img_path.startswith("images/"):
            img_path = img_path[len("images/"):]
        
        full_path = os.path.join(self.img_root, img_path)
        image = Image.open(full_path).convert("RGB")
        image = self.transform(image)
        image = image[self.channel_indices, :, :]

        return image, int(label)


def train_and_evaluate_channel_modes(model, device, num_epochs=20):
    """Train and evaluate model on different channel combinations"""
    results = {}
    
    for channel_mode in ["RGB", "RG", "GB", "RB", "R", "G", "B"]:
        print(f"\n{'='*50}")
        print(f"Training with channel mode: {channel_mode}")
        print(f"{'='*50}")
        
        # Create datasets
        train_dataset = MiniImageNetDataset('./train.txt', './images', channel_mode=channel_mode)
        val_dataset = MiniImageNetDataset('./val.txt', './images', channel_mode=channel_mode)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        
        # Reset model
        if isinstance(model, FlexibleCNN):
            model = FlexibleCNN(max_channels=3, num_classes=100).to(device)
        else:
            num_channels = len(CHANNEL_MAP[channel_mode])
            model = StandardCNN(in_channels=num_channels, num_classes=100).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        best_val_acc = 0.0
        train_losses = []
        val_accs = []
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100.*correct/total
            
            # Validate
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            val_acc = 100.*correct/total
            val_accs.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        results[channel_mode] = {
            'best_val_acc': best_val_acc,
            'final_val_acc': val_acc,
            'val_accs': val_accs
        }
        
        print(f"Best validation accuracy for {channel_mode}: {best_val_acc:.2f}%")
    
    return results


def plot_channel_comparison(results, save_path='channel_comparison.png'):
    """Plot comparison of different channel modes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot of best accuracies
    channels = list(results.keys())
    best_accs = [results[ch]['best_val_acc'] for ch in channels]
    
    bars = ax1.bar(channels, best_accs)
    ax1.set_xlabel('Channel Mode')
    ax1.set_ylabel('Best Validation Accuracy (%)')
    ax1.set_title('Performance Comparison Across Channel Modes')
    
    # Color bars based on number of channels
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'black']
    for bar, ch, color in zip(bars, channels, colors):
        if len(ch) == 1:
            bar.set_color(color)
        elif len(ch) == 2:
            bar.set_color('orange')
        else:
            bar.set_color('purple')
    
    # Add value labels on bars
    for bar, acc in zip(bars, best_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Learning curves
    for ch, data in results.items():
        ax2.plot(data['val_accs'], label=ch, linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Learning Curves for Different Channel Modes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison plot to {save_path}")
    plt.show()


def main():
    """Main function for Task A"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test dynamic convolution model
    print("\n🔧 Testing Dynamic Convolution Model")
    dynamic_model = FlexibleCNN(max_channels=3, num_classes=100).to(device)
    
    # Test with different input channels
    for channels in [1, 2, 3]:
        dummy_input = torch.randn(1, channels, 128, 128).to(device)
        output = dynamic_model(dummy_input)
        print(f"Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
    
    # Train and evaluate on all channel modes
    print("\n🚀 Starting comprehensive channel evaluation...")
    results = train_and_evaluate_channel_modes(dynamic_model, device, num_epochs=30)
    
    # Plot results
    plot_channel_comparison(results)
    
    # Summary
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    for ch, data in results.items():
        print(f"{ch}: Best Val Acc = {data['best_val_acc']:.2f}%")
    
    # Find best channel mode
    best_channel = max(results.items(), key=lambda x: x[1]['best_val_acc'])
    print(f"\n✅ Best channel mode: {best_channel[0]} with {best_channel[1]['best_val_acc']:.2f}% accuracy")


if __name__ == "__main__":
    main()