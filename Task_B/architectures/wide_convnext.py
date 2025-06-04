"""
Wide ConvNeXt - Second best performing architecture
Achieves 44.67% accuracy with only 1.8M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WideConvNeXt(nn.Module):
    """Wide ConvNeXt - High performance with low parameter count"""
    def __init__(self, dim=192, num_classes=100):
        super().__init__()
        
        # Layer 1: Patchify stem (7x7 conv)
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(dim)
        )
        
        # Layer 2-3: ConvNeXt blocks
        self.block1 = ConvNeXtBlock(dim, dim * 2)
        self.block2 = ConvNeXtBlock(dim * 2, dim * 2)
        
        # Layer 4: Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with depthwise conv"""
    def __init__(self, dim_in, dim_out, expansion=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim_in, dim_in, kernel_size=7, padding=3, groups=dim_in)
        self.norm = nn.BatchNorm2d(dim_in)
        self.pwconv1 = nn.Conv2d(dim_in, dim_in * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim_in * expansion, dim_out, kernel_size=1)
        self.shortcut = nn.Conv2d(dim_in, dim_out, kernel_size=1) if dim_in != dim_out else nn.Identity()
        
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x + self.shortcut(residual)
        return x