"""
Evaluation metrics and utilities for model performance analysis
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name=''):
        self.name = name
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricTracker:
    """Tracks multiple metrics during training"""
    
    def __init__(self, metrics=['loss', 'acc', 'top5_acc']):
        self.metrics = metrics
        self.history = defaultdict(list)
        self.current_epoch = defaultdict(lambda: AverageMeter())
        
    def update(self, metric_dict, n=1):
        """Update metrics with new values"""
        for key, value in metric_dict.items():
            if key in self.metrics:
                self.current_epoch[key].update(value, n)
    
    def epoch_end(self):
        """Store epoch averages and reset"""
        for metric in self.metrics:
            if metric in self.current_epoch:
                self.history[metric].append(self.current_epoch[metric].avg)
        
        # Reset for next epoch
        self.current_epoch.clear()
    
    def get_history(self):
        """Get training history"""
        return dict(self.history)
    
    def plot_history(self, save_path=None):
        """Plot training history"""
        n_metrics = len(self.history)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (metric, values) in enumerate(self.history.items()):
            axes[idx].plot(values, linewidth=2)
            axes[idx].set_title(f'{metric.capitalize()} History')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions
    
    Args:
        output: Model predictions (B, num_classes)
        target: True labels (B,)
        topk: Tuple of k values for top-k accuracy
    
    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


def compute_confusion_matrix(y_true, y_pred, num_classes=100, normalize=True):
    """
    Compute confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        normalize: Whether to normalize the matrix
    
    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero
    
    return cm


def plot_confusion_matrix(cm, class_names=None, save_path=None, figsize=(10, 8)):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Use a subset of the matrix if too large
    if cm.shape[0] > 20:
        # Show only first 20 classes for visibility
        cm = cm[:20, :20]
        if class_names:
            class_names = class_names[:20]
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if cm.dtype == float else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if cm.dtype == float else 'Count'}
    )
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def evaluate(self, dataloader, criterion=None):
        """
        Evaluate model on a dataset
        
        Args:
            dataloader: DataLoader for evaluation
            criterion: Loss function (optional)
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        total_samples = 0
        
        # Timing
        start_time = time.time()
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss if criterion provided
                if criterion:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * inputs.size(0)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Store for metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_samples += inputs.size(0)
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = (all_preds == all_labels).mean() * 100
        avg_loss = total_loss / total_samples if criterion else 0
        eval_time = time.time() - start_time
        
        # Per-class accuracy
        per_class_acc = {}
        for class_idx in range(len(np.unique(all_labels))):
            mask = all_labels == class_idx
            if mask.sum() > 0:
                per_class_acc[class_idx] = (all_preds[mask] == all_labels[mask]).mean() * 100
        
        results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'eval_time': eval_time,
            'samples_per_second': total_samples / eval_time,
            'per_class_accuracy': per_class_acc,
            'predictions': all_preds,
            'labels': all_labels
        }
        
        return results
    
    def generate_report(self, results, class_names=None):
        """Generate evaluation report"""
        print("="*60)
        print("Model Evaluation Report")
        print("="*60)
        print(f"Overall Accuracy: {results['accuracy']:.2f}%")
        print(f"Average Loss: {results['loss']:.4f}")
        print(f"Evaluation Time: {results['eval_time']:.2f}s")
        print(f"Throughput: {results['samples_per_second']:.1f} samples/s")
        
        # Top-5 and Bottom-5 classes
        per_class = results['per_class_accuracy']
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 5 Classes:")
        for i, (class_idx, acc) in enumerate(sorted_classes[:5]):
            class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
            print(f"  {i+1}. {class_name}: {acc:.2f}%")
        
        print("\nBottom 5 Classes:")
        for i, (class_idx, acc) in enumerate(sorted_classes[-5:]):
            class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
            print(f"  {i+1}. {class_name}: {acc:.2f}%")
        
        print("="*60)


def calculate_model_stats(model):
    """
    Calculate model statistics
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model statistics
    """
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    # Calculate model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'params_millions': total_params / 1e6
    }


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics utilities...")
    
    # Test accuracy computation
    output = torch.randn(32, 100)
    target = torch.randint(0, 100, (32,))
    
    top1, top5 = accuracy(output, target, topk=(1, 5))
    print(f"Top-1 Accuracy: {top1.item():.2f}%")
    print(f"Top-5 Accuracy: {top5.item():.2f}%")
    
    # Test metric tracker
    tracker = MetricTracker(['loss', 'acc'])
    for epoch in range(5):
        # Simulate epoch
        for i in range(10):
            tracker.update({
                'loss': np.random.random(),
                'acc': np.random.random() * 100
            })
        tracker.epoch_end()
    
    history = tracker.get_history()
    print(f"\nTraining history: {history}")
    
    print("\n✅ Metrics utilities test complete!")