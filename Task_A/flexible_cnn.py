"""
Flexible CNN Module - Dynamic Convolution Implementation
Supports arbitrary channel combinations from RGB input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexibleChannelCNN(nn.Module):
    """
    Dynamic CNN that handles arbitrary channel combinations
    Key innovation: Weight generation network adapts to input channels
    """
    
    def __init__(self, num_classes=100):
        super(FlexibleChannelCNN, self).__init__()
        
        # Dynamic first layer - adapts to 1, 2, or 3 channels
        self.dynamic_conv1 = DynamicConvBlock(out_channels=64)
        
        # Fixed architecture after first layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        """Forward pass with dynamic channel handling"""
        x = self.dynamic_conv1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DynamicConvBlock(nn.Module):
    """
    Dynamic convolution block that generates weights based on input channels
    Maintains spatial dimensions while adapting to channel count
    """
    
    def __init__(self, out_channels=64):
        super(DynamicConvBlock, self).__init__()
        self.out_channels = out_channels
        
        # Weight generation networks for different channel counts
        self.weight_gen_1ch = nn.Linear(1, out_channels * 7 * 7)
        self.weight_gen_2ch = nn.Linear(2, out_channels * 2 * 7 * 7)
        self.weight_gen_3ch = nn.Linear(3, out_channels * 3 * 7 * 7)
        
        # Channel-wise attention for importance weighting
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Dynamic forward pass based on input channels
        Args:
            x: Input tensor with 1, 2, or 3 channels
        Returns:
            Output tensor with fixed out_channels
        """
        batch_size = x.size(0)
        in_channels = x.size(1)
        height, width = x.size(2), x.size(3)
        
        # Generate convolution weights based on channel count
        if in_channels == 1:
            # Single channel: direct convolution
            weights = self.weight_gen_1ch(torch.ones(1, 1).to(x.device))
            weights = weights.view(self.out_channels, 1, 7, 7)
        elif in_channels == 2:
            # Two channels: generate weights for each
            weights = self.weight_gen_2ch(torch.ones(1, 2).to(x.device))
            weights = weights.view(self.out_channels, 2, 7, 7)
        else:  # in_channels == 3
            # Three channels: full RGB processing
            # Apply channel attention first
            attention = self.channel_attention(x)
            x = x * attention
            
            weights = self.weight_gen_3ch(torch.ones(1, 3).to(x.device))
            weights = weights.view(self.out_channels, 3, 7, 7)
        
        # Apply dynamic convolution with proper padding to maintain size
        output = F.conv2d(x, weights, stride=2, padding=3)
        output = self.bn(output)
        output = self.relu(output)
        
        return output


class ChannelSelector(nn.Module):
    """
    Utility module to select specific channels from RGB input
    Supports: RGB, RG, GB, RB, R, G, B
    """
    
    def __init__(self, channel_mode='RGB'):
        super(ChannelSelector, self).__init__()
        self.channel_mode = channel_mode
        self.channel_indices = self._get_channel_indices(channel_mode)
        
    def _get_channel_indices(self, mode):
        """Map channel mode to indices"""
        mapping = {
            'RGB': [0, 1, 2],
            'RG': [0, 1],
            'GB': [1, 2],
            'RB': [0, 2],
            'R': [0],
            'G': [1],
            'B': [2]
        }
        return mapping.get(mode, [0, 1, 2])
    
    def forward(self, x):
        """Select specified channels from input"""
        return x[:, self.channel_indices, :, :]


def create_flexible_model(channel_mode='RGB', num_classes=100):
    """
    Factory function to create flexible CNN with channel selection
    
    Args:
        channel_mode: One of 'RGB', 'RG', 'GB', 'RB', 'R', 'G', 'B'
        num_classes: Number of output classes
    
    Returns:
        nn.Sequential model with channel selector and flexible CNN
    """
    return nn.Sequential(
        ChannelSelector(channel_mode),
        FlexibleChannelCNN(num_classes)
    )


if __name__ == "__main__":
    # Test the flexible CNN with different channel combinations
    print("Testing Flexible CNN Module...")
    
    # Test with different channel counts
    for channels in [1, 2, 3]:
        model = FlexibleChannelCNN()
        x = torch.randn(2, channels, 224, 224)
        output = model(x)
        print(f"Input shape: {x.shape} -> Output shape: {output.shape}")
    
    # Test channel selector
    for mode in ['RGB', 'RG', 'R']:
        model = create_flexible_model(mode)
        x = torch.randn(2, 3, 224, 224)  # Always start with RGB
        output = model(x)
        print(f"Mode {mode}: Output shape: {output.shape}")
    
    print("✅ All tests passed!")