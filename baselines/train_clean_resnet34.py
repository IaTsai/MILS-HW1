#!/usr/bin/env python3
"""
Clean ResNet34 training script for establishing correct baseline
Target: 60-70% validation accuracy on mini-ImageNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class MiniImageNetDataset(torch.utils.data.Dataset):
    """Clean dataset implementation"""
    def __init__(self, txt_path, img_root, transform=None):
        with open(txt_path, 'r') as f:
            self.samples = [line.strip().split() for line in f.readlines()]
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Handle path format
        if img_path.startswith("images/"):
            img_path = img_path[len("images/"):]
        
        full_path = os.path.join(self.img_root, img_path)
        
        try:
            image = Image.open(full_path).convert("RGB")
        except:
            # Return a black image if file not found
            print(f"Warning: Could not load {full_path}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, int(label)


def get_transforms():
    """Standard ImageNet transforms"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss/len(dataloader),
            'acc': 100.*correct/total
        })
    
    return running_loss/len(dataloader), 100.*correct/total


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'acc': 100.*correct/total})
    
    return running_loss/len(dataloader), 100.*correct/total


def train_resnet34(num_epochs=50):
    """Main training function"""
    print("="*60)
    print("Training Clean ResNet34 Baseline")
    print("="*60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data setup
    train_transform, val_transform = get_transforms()
    
    print("\n📁 Loading datasets...")
    train_dataset = MiniImageNetDataset('./train.txt', './images', transform=train_transform)
    val_dataset = MiniImageNetDataset('./val.txt', './images', transform=val_transform)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256,  # Larger batch for ResNet
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Model setup
    print("\n🏗️ Creating ResNet34 model...")
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 100)  # 100 classes
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # SGD with momentum - standard for ResNet
    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Learning rate schedule
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[30, 40], 
        gamma=0.1
    )
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    # Training loop
    print("\n🚀 Starting training...")
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'train_loss': train_loss,
            }
            torch.save(checkpoint, 'clean_resnet34_best.pth')
            print(f"✅ Saved best model with Val Acc: {val_acc:.2f}%")
        
        # Early stopping if we reach good performance
        if val_acc >= 65.0:
            print(f"\n🎯 Reached target performance of 65%+")
            break
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ResNet34 - Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc', linewidth=2)
    plt.plot(val_accs, label='Val Acc', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('ResNet34 - Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clean_resnet34_training.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved training curves to clean_resnet34_training.png")
    
    # Final verification
    print("\n🔍 Final Model Verification...")
    model.load_state_dict(checkpoint['model_state_dict'])
    final_val_loss, final_val_acc = validate(model, val_loader, criterion, device)
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    
    return best_val_acc


if __name__ == "__main__":
    # Run training
    best_acc = train_resnet34(num_epochs=50)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best ResNet34 Validation Accuracy: {best_acc:.2f}%")
    print(f"Target for Task B (90%): {best_acc * 0.9:.2f}%")
    print("="*60)