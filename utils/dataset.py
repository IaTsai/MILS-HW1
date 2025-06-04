"""
Dataset utilities for mini-ImageNet loading and preprocessing
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class MiniImageNetDataset(Dataset):
    """
    Mini-ImageNet dataset loader
    Supports flexible channel selection for Task A
    """
    
    def __init__(self, txt_path, img_root='images', transform=None, channel_mode='RGB'):
        """
        Args:
            txt_path: Path to train.txt or val.txt
            img_root: Root directory for images
            transform: Torchvision transforms
            channel_mode: One of 'RGB', 'RG', 'GB', 'RB', 'R', 'G', 'B'
        """
        self.img_root = img_root
        self.transform = transform
        self.channel_mode = channel_mode
        self.channel_indices = self._get_channel_indices(channel_mode)
        
        # Load samples from text file
        self.samples = []
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        img_path = parts[0]
                        label = int(parts[1])
                        self.samples.append((img_path, label))
        
        print(f"Loaded {len(self.samples)} samples from {txt_path}")
        
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Handle different path formats
        if not os.path.isabs(img_path):
            if img_path.startswith('images/'):
                img_path = img_path[7:]  # Remove 'images/' prefix
            img_path = os.path.join(self.img_root, img_path)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return black image as fallback
            image = Image.new('RGB', (224, 224))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Select specified channels
        if self.channel_mode != 'RGB':
            image = image[self.channel_indices, :, :]
        
        return image, label


def get_data_transforms(is_training=True, input_size=224):
    """
    Get data transforms for training or validation
    
    Args:
        is_training: Whether to apply training augmentations
        input_size: Input image size
    
    Returns:
        torchvision.transforms.Compose object
    """
    
    if is_training:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2
            ),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform


def get_data_loaders(
    train_txt='train.txt',
    val_txt='val.txt',
    img_root='images',
    batch_size=32,
    num_workers=4,
    channel_mode='RGB',
    pin_memory=True
):
    """
    Create train and validation data loaders
    
    Args:
        train_txt: Path to training file list
        val_txt: Path to validation file list
        img_root: Root directory for images
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        channel_mode: Channel selection mode
        pin_memory: Whether to pin memory for GPU
    
    Returns:
        train_loader, val_loader
    """
    
    # Get transforms
    train_transform = get_data_transforms(is_training=True)
    val_transform = get_data_transforms(is_training=False)
    
    # Create datasets
    train_dataset = MiniImageNetDataset(
        train_txt,
        img_root=img_root,
        transform=train_transform,
        channel_mode=channel_mode
    )
    
    val_dataset = MiniImageNetDataset(
        val_txt,
        img_root=img_root,
        transform=val_transform,
        channel_mode=channel_mode
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Can use larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )
    
    return train_loader, val_loader


class CutMix:
    """CutMix augmentation for better generalization"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, images, targets):
        """
        Apply CutMix augmentation
        
        Args:
            images: Batch of images (B, C, H, W)
            targets: Batch of labels (B,)
            
        Returns:
            mixed_images, targets_a, targets_b, lam
        """
        batch_size = images.size(0)
        indices = torch.randperm(batch_size).to(images.device)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get random box coordinates
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
        # Apply CutMix
        images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        
        return images, targets, targets[indices], lam
    
    def _rand_bbox(self, size, lam):
        """Generate random bounding box"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Random position
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset utilities...")
    
    # Test with different channel modes
    for mode in ['RGB', 'RG', 'R']:
        print(f"\nTesting {mode} mode:")
        dataset = MiniImageNetDataset(
            'val.txt',
            channel_mode=mode,
            transform=get_data_transforms(is_training=False)
        )
        
        # Load one sample
        if len(dataset) > 0:
            image, label = dataset[0]
            print(f"Image shape: {image.shape}")
            print(f"Label: {label}")
    
    print("\n✅ Dataset utilities test complete!")