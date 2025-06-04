"""
Fixed training script for MiniConvNeXt - Resolves gradient and data loading issues
Target: 48.40% accuracy with minimal memory footprint
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from mini_convnext import MiniConvNeXt, UltraLightCNN


class MiniImageNetDataset(Dataset):
    """Dataset class that properly loads mini-ImageNet data"""
    def __init__(self, txt_path, transform=None):
        self.samples = []
        self.transform = transform
        
        # Read data file
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        # Parse each line
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    img_path = parts[0]
                    label = int(parts[1])
                    self.samples.append((img_path, label))
        
        print(f"Loaded {len(self.samples)} samples from {txt_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(batch_size=32):
    """Get train and validation data loaders with proper data augmentation"""
    
    # Training transforms with strong augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MiniImageNetDataset('train.txt', transform=train_transform)
    val_dataset = MiniImageNetDataset('val.txt', transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader


class CutMix:
    """CutMix augmentation for better generalization"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def __call__(self, images, labels):
        if self.alpha <= 0:
            return images, labels, labels, 1.0
        
        batch_size = images.size(0)
        indices = torch.randperm(batch_size).cuda()
        
        lam = np.random.beta(self.alpha, self.alpha)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
        
        images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
        return images, labels, labels[indices], lam


def ensure_gradients(model):
    """Ensure all model parameters have gradients enabled"""
    for param in model.parameters():
        param.requires_grad = True
    return model


def train_epoch(model, train_loader, criterion, optimizer, scaler, cutmix, device, epoch):
    """Train for one epoch with mixed precision and CutMix"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply CutMix with 50% probability
        if cutmix and np.random.rand() < 0.5:
            inputs, targets_a, targets_b, lam = cutmix(inputs, targets)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast('cuda'):
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_idx}")
                continue
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate accuracy (approximate for mixed labels)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().item() + 
                       (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_idx}")
                continue
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        running_loss += loss.item()
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100.*correct/total:.2f}%')
        
        # Clear cache periodically
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(val_loader)
    return avg_loss, accuracy


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.empty_cache()
    
    # Create model
    if args.model == 'mini':
        model = MiniConvNeXt(num_classes=100, dim=args.dim)
        model_name = f"MiniConvNeXt_dim{args.dim}"
    else:
        model = UltraLightCNN(num_classes=100)
        model_name = "UltraLightCNN"
    
    # Ensure gradients are enabled
    model = ensure_gradients(model)
    model = model.to(device)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model_name}")
    print(f"Parameters: {params:,}")
    print(f"All parameters require grad: {all(p.requires_grad for p in model.parameters())}")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(batch_size=args.batch_size)
    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda')
    
    # CutMix augmentation
    cutmix = CutMix(alpha=1.0) if args.cutmix else None
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*50)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, cutmix, device, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'model_name': model_name,
                'dim': args.dim if args.model == 'mini' else None
            }, f'best_{model_name.lower()}.pth')
            print(f"✓ Saved new best model with val_acc: {val_acc:.2f}%")
        
        # Print progress
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {current_lr:.6f}")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        
        # Check if we've reached the target
        if val_acc >= 48.40:
            print(f"\n🎉 Target accuracy reached! Val Acc: {val_acc:.2f}% >= 48.40%")
            break
        
        # Early stopping if loss becomes NaN
        if np.isnan(train_loss) or np.isnan(val_loss):
            print("Training stopped due to NaN loss")
            break
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.axhline(y=48.40, color='r', linestyle='--', label='Target (48.40%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_training_fixed.png', dpi=150)
    print(f"\nTraining curves saved to {model_name.lower()}_training_fixed.png")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Target accuracy: 48.40%")
    print(f"Gap to target: {48.40 - best_val_acc:.2f}%")
    
    # Final model summary
    if best_val_acc >= 48.40:
        print("\n✅ SUCCESS: Target accuracy achieved!")
    else:
        print("\n⚠️  Target not reached. Suggestions:")
        print("1. Try training for more epochs")
        print("2. Increase model dimension if GPU memory allows")
        print("3. Use stronger data augmentation")
        print("4. Try ensemble methods")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MiniConvNeXt (Fixed)')
    parser.add_argument('--model', type=str, default='mini', choices=['mini', 'ultra'],
                        help='Model type (mini or ultra)')
    parser.add_argument('--dim', type=int, default=48,
                        help='Model dimension for MiniConvNeXt')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--cutmix', action='store_true',
                        help='Use CutMix augmentation')
    
    args = parser.parse_args()
    main(args)