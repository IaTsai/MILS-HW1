"""
10種高潛力4層架構的實現
每個架構都針對mini-ImageNet 100類分類任務優化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 1. Wide ConvNeXt (高潛力 ⭐⭐⭐⭐⭐)
class WideConvNeXt(nn.Module):
    """基於ConvNeXt但大幅增加寬度的4層網路"""
    def __init__(self, dim=256, num_classes=100):
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


# 2. ResNeSt-4Layer (高潛力 ⭐⭐⭐⭐⭐)
class ResNeSt4Layer(nn.Module):
    """帶有Split Attention的4層網路"""
    def __init__(self, dim=128, num_classes=100):
        super().__init__()
        
        # Layer 1: Conv stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2-3: ResNeSt blocks with split attention
        self.block1 = ResNeStBlock(dim, dim * 2, stride=2)
        self.block2 = ResNeStBlock(dim * 2, dim * 4, stride=2)
        
        # Layer 4: Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim * 4, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x


class ResNeStBlock(nn.Module):
    """ResNeSt block with split attention"""
    def __init__(self, in_channels, out_channels, stride=1, radix=2, cardinality=1):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels * radix, 3, padding=1, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(out_channels * radix)
        
        self.fc1 = nn.Conv2d(out_channels, out_channels // 4, 1)
        self.fc2 = nn.Conv2d(out_channels // 4, out_channels * radix, 1)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = x
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Split attention
        batch, rchannel = x.shape[:2]
        splits = x.view(batch, self.radix, self.channels, x.shape[2], x.shape[3])
        gap = splits.sum(dim=1).mean(dim=[2, 3], keepdim=True)
        
        atten = F.relu(self.fc1(gap))
        atten = torch.sigmoid(self.fc2(atten))
        atten = atten.view(batch, self.radix, self.channels, 1, 1)
        
        out = (splits * atten).sum(dim=1)
        out = out + self.shortcut(residual)
        
        return F.relu(out)


# 3. Mini Swin Transformer (高潛力 ⭐⭐⭐⭐)
class MiniSwinTransformer(nn.Module):
    """迷你版Swin Transformer - 4層"""
    def __init__(self, embed_dim=96, num_classes=100):
        super().__init__()
        
        # Layer 1: Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.patch_norm = nn.LayerNorm(embed_dim)
        
        # Layer 2: Swin Block (W-MSA)
        self.block1 = SwinBlock(embed_dim, num_heads=3, window_size=7, shift=False)
        
        # Layer 3: Swin Block (SW-MSA)
        self.block2 = SwinBlock(embed_dim, num_heads=3, window_size=7, shift=True)
        
        # Layer 4: MLP Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # B, C, H, W
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, HW, C
        x = self.patch_norm(x)
        
        # Swin blocks
        x = self.block1(x, H, W)
        x = self.block2(x, H, W)
        
        # Reshape back and classify
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.head(x)
        return x


class SwinBlock(nn.Module):
    """Simplified Swin Transformer block"""
    def __init__(self, dim, num_heads, window_size=7, shift=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift = shift
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        
    def forward(self, x, H, W):
        # Window-based self-attention
        shortcut = x
        x = self.norm1(x)
        
        # Reshape for windowing
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        # Cyclic shift if needed
        if self.shift:
            x = torch.roll(x, shifts=(-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
        
        # Simplified window attention (placeholder)
        x = x.view(B, L, C)
        x = self.attn(x)
        x = shortcut + x
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class WindowAttention(nn.Module):
    """Simplified window attention"""
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# 4. EfficientNet-Style (高潛力 ⭐⭐⭐⭐)
class EfficientNetStyle(nn.Module):
    """基於EfficientNet的4層網路"""
    def __init__(self, width_mult=1.5, num_classes=100):
        super().__init__()
        base_channels = int(32 * width_mult)
        
        # Layer 1: Conv stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True)
        )
        
        # Layer 2-3: MBConv blocks with SE
        self.block1 = MBConvBlock(base_channels, base_channels * 2, stride=2, expand_ratio=6)
        self.block2 = MBConvBlock(base_channels * 2, base_channels * 4, stride=2, expand_ratio=6)
        
        # Layer 4: Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck with SE"""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise conv
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])
        
        # SE layer
        layers.append(SEBlock(hidden_dim))
        
        # Projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 5. Wide ResNet Block (高潛力 ⭐⭐⭐⭐)
class WideResNet4Layer(nn.Module):
    """寬度優先的4層ResNet"""
    def __init__(self, width=4, num_classes=100):
        super().__init__()
        base_channels = 64
        
        # Layer 1: Conv stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels * width, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels * width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Layer 2-3: Wide ResNet blocks
        self.block1 = WideResBlock(base_channels * width, base_channels * width * 2, stride=2)
        self.block2 = WideResBlock(base_channels * width * 2, base_channels * width * 4, stride=2)
        
        # Layer 4: Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * width * 4, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x


class WideResBlock(nn.Module):
    """Wide residual block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


# 6. Attention-CNN Hybrid (高潛力 ⭐⭐⭐⭐)
class AttentionCNNHybrid(nn.Module):
    """CNN與注意力機制的混合架構"""
    def __init__(self, dim=128, num_classes=100):
        super().__init__()
        
        # Layer 1: Conv stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2: Conv + Self-Attention
        self.block1 = HybridBlock(dim, dim * 2, use_self_attention=True)
        
        # Layer 3: Conv + Channel Attention
        self.block2 = HybridBlock(dim * 2, dim * 4, use_channel_attention=True)
        
        # Layer 4: Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim * 4, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x


class HybridBlock(nn.Module):
    """混合卷積和注意力的block"""
    def __init__(self, in_channels, out_channels, use_self_attention=False, use_channel_attention=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        if use_self_attention:
            self.attention = SelfAttention2D(out_channels)
        elif use_channel_attention:
            self.attention = ChannelAttention(out_channels)
        else:
            self.attention = nn.Identity()
            
    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
        return x


class SelfAttention2D(nn.Module):
    """2D self-attention"""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        return self.gamma * out + x


class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = self.fc(avg_y) + self.fc(max_y)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y


# 7. ConvMixer-Style (創新 ⭐⭐⭐⭐)
class ConvMixerStyle(nn.Module):
    """基於ConvMixer概念的4層網路"""
    def __init__(self, dim=256, num_classes=100):
        super().__init__()
        
        # Layer 1: Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=7, stride=7),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        
        # Layer 2-3: ConvMixer blocks
        self.block1 = ConvMixerBlock(dim, kernel_size=9)
        self.block2 = ConvMixerBlock(dim, kernel_size=7)
        
        # Layer 4: Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x


class ConvMixerBlock(nn.Module):
    """ConvMixer block"""
    def __init__(self, dim, kernel_size=9):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size//2)
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = x + residual
        
        x = self.pointwise(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x


# 8. Multi-Scale CNN (高潛力 ⭐⭐⭐⭐)
class MultiScaleCNN(nn.Module):
    """多尺度特徵融合的4層網路"""
    def __init__(self, dim=96, num_classes=100):
        super().__init__()
        
        # Layer 1: Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2: Multi-scale conv
        self.multi_scale1 = MultiScaleBlock(dim, dim * 2)
        
        # Layer 3: Multi-scale conv with fusion
        self.multi_scale2 = MultiScaleBlock(dim * 2, dim * 4)
        
        # Layer 4: Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim * 4, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.multi_scale1(x)
        x = self.multi_scale2(x)
        x = self.head(x)
        return x


class MultiScaleBlock(nn.Module):
    """多尺度卷積塊"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4
        
        # 1x1 branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, stride=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 5x5 branch (implemented as two 3x3)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Pool branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return out


# 9. Dense-Efficient Net (高潛力 ⭐⭐⭐)
class DenseEfficientNet(nn.Module):
    """Dense連接與效率優化的4層網路"""
    def __init__(self, growth_rate=32, num_classes=100):
        super().__init__()
        
        # Layer 1: Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, growth_rate * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(growth_rate * 2),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2-3: Dense blocks
        self.dense1 = DenseBlock(growth_rate * 2, growth_rate, num_layers=4)
        self.trans1 = TransitionBlock(growth_rate * 2 + growth_rate * 4, growth_rate * 4)
        
        self.dense2 = DenseBlock(growth_rate * 4, growth_rate, num_layers=4)
        self.trans2 = TransitionBlock(growth_rate * 4 + growth_rate * 4, growth_rate * 8)
        
        # Layer 4: Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(growth_rate * 8, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.head(x)
        return x


class DenseBlock(nn.Module):
    """Dense block"""
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                DenseLayer(in_channels + i * growth_rate, growth_rate)
            )
            
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class DenseLayer(nn.Module):
    """Dense layer"""
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, growth_rate * 4, 1)
        self.bn2 = nn.BatchNorm2d(growth_rate * 4)
        self.conv2 = nn.Conv2d(growth_rate * 4, growth_rate, 3, padding=1)
        
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out


class TransitionBlock(nn.Module):
    """Transition block for DenseNet"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.pool = nn.AvgPool2d(2)
        
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


# 10. Ghost-Net Style (輕量高效 ⭐⭐⭐)
class GhostNetStyle(nn.Module):
    """基於Ghost convolution的輕量化4層網路"""
    def __init__(self, width=1.5, num_classes=100):
        super().__init__()
        channels = [int(16 * width), int(24 * width), int(40 * width), int(80 * width)]
        
        # Layer 1: Conv stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2-3: Ghost bottlenecks
        self.ghost1 = GhostBottleneck(channels[0], channels[1], channels[2], stride=2)
        self.ghost2 = GhostBottleneck(channels[2], channels[2], channels[3], stride=2)
        
        # Layer 4: Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[3], num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.ghost1(x)
        x = self.ghost2(x)
        x = self.head(x)
        return x


class GhostBottleneck(nn.Module):
    """Ghost bottleneck"""
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()
        self.ghost1 = GhostModule(in_channels, mid_channels)
        
        if stride == 2:
            self.conv_dw = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, 
                                   padding=1, groups=mid_channels)
            self.bn_dw = nn.BatchNorm2d(mid_channels)
        else:
            self.conv_dw = None
            
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)
        
        if stride == 2 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = x
        
        x = self.ghost1(x)
        
        if self.conv_dw is not None:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
            
        x = self.ghost2(x)
        x = x + self.shortcut(residual)
        
        return x


class GhostModule(nn.Module):
    """Ghost module"""
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, relu=True):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )
        
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]