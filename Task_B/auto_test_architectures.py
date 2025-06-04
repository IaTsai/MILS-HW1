#!/usr/bin/env python3
"""
自動化架構測試系統 - 測試10種高潛力的4層網路架構
目標: 找出最有希望達到ResNet34 90%性能(48.40%)的架構
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Import dataset class (will reuse from existing code)
from train_mini_convnext_fixed import MiniImageNetDataset, get_data_loaders

# Import all architectures
from architecture_implementations import (
    WideConvNeXt, ResNeSt4Layer, MiniSwinTransformer, 
    EfficientNetStyle, WideResNet4Layer, AttentionCNNHybrid,
    ConvMixerStyle, MultiScaleCNN, DenseEfficientNet, GhostNetStyle
)


class ArchitectureTester:
    """自動化架構測試器"""
    
    def __init__(self, save_dir="architecture_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def optimize_config_for_arch(self, arch_name):
        """根據架構類型自動選擇最佳配置"""
        configs = {
            "wide_convnext": {
                "dim": 192,  # 更寬但不至於OOM
                "batch_size": 16,
                "lr": 3e-4,
                "weight_decay": 0.05
            },
            "resnest_4layer": {
                "dim": 128,
                "batch_size": 24,
                "lr": 4e-4,
                "weight_decay": 0.01
            },
            "mini_swin": {
                "embed_dim": 96,
                "batch_size": 32,
                "lr": 5e-4,
                "weight_decay": 0.05
            },
            "efficientnet_style": {
                "width_mult": 1.5,
                "batch_size": 32,
                "lr": 3e-4,
                "weight_decay": 0.01
            },
            "wide_resnet": {
                "width": 4,
                "batch_size": 24,
                "lr": 3e-4,
                "weight_decay": 0.05
            },
            "attention_cnn": {
                "dim": 128,
                "batch_size": 28,
                "lr": 4e-4,
                "weight_decay": 0.01
            },
            "convmixer_style": {
                "dim": 256,
                "batch_size": 32,
                "lr": 5e-4,
                "weight_decay": 0.01
            },
            "multiscale_cnn": {
                "dim": 96,
                "batch_size": 24,
                "lr": 3e-4,
                "weight_decay": 0.05
            },
            "dense_efficient": {
                "growth_rate": 32,
                "batch_size": 28,
                "lr": 4e-4,
                "weight_decay": 0.01
            },
            "ghostnet_style": {
                "width": 1.5,
                "batch_size": 48,
                "lr": 4e-4,
                "weight_decay": 0.01
            }
        }
        return configs.get(arch_name, {"dim": 128, "batch_size": 24, "lr": 3e-4, "weight_decay": 0.01})
    
    def create_architecture(self, arch_name, config):
        """創建指定的架構"""
        if arch_name == "wide_convnext":
            return WideConvNeXt(dim=config['dim'])
        elif arch_name == "resnest_4layer":
            return ResNeSt4Layer(dim=config['dim'])
        elif arch_name == "mini_swin":
            return MiniSwinTransformer(embed_dim=config['embed_dim'])
        elif arch_name == "efficientnet_style":
            return EfficientNetStyle(width_mult=config['width_mult'])
        elif arch_name == "wide_resnet":
            return WideResNet4Layer(width=config['width'])
        elif arch_name == "attention_cnn":
            return AttentionCNNHybrid(dim=config['dim'])
        elif arch_name == "convmixer_style":
            return ConvMixerStyle(dim=config['dim'])
        elif arch_name == "multiscale_cnn":
            return MultiScaleCNN(dim=config['dim'])
        elif arch_name == "dense_efficient":
            return DenseEfficientNet(growth_rate=config['growth_rate'])
        elif arch_name == "ghostnet_style":
            return GhostNetStyle(width=config['width'])
        else:
            raise ValueError(f"Unknown architecture: {arch_name}")
    
    def train_architecture(self, arch_name, epochs=10):
        """訓練單個架構"""
        print(f"\n{'='*60}")
        print(f"Testing Architecture: {arch_name}")
        print(f"{'='*60}")
        
        # 獲取最佳配置
        config = self.optimize_config_for_arch(arch_name)
        print(f"Config: {config}")
        
        # 創建模型
        model = self.create_architecture(arch_name, config).to(self.device)
        
        # 計算參數量
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {num_params:,}")
        
        # 獲取數據載入器
        train_loader, val_loader = get_data_loaders(batch_size=config['batch_size'])
        
        # 設置優化器和調度器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler = GradScaler()
        
        # 訓練歷史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        best_val_acc = 0.0
        start_time = time.time()
        
        # 訓練循環
        for epoch in range(epochs):
            # 訓練階段
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, scaler
            )
            
            # 驗證階段
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # 更新學習率
            scheduler.step()
            
            # 記錄歷史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # 更新最佳準確率
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 
                          os.path.join(self.save_dir, f"{arch_name}_best.pth"))
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train: {train_acc:.2f}% - Val: {val_acc:.2f}% - "
                  f"Best: {best_val_acc:.2f}%")
        
        # 計算訓練時間
        train_time = time.time() - start_time
        
        # 保存結果
        result = {
            'architecture': arch_name,
            'config': config,
            'num_params': num_params,
            'best_val_acc': best_val_acc,
            'final_val_acc': val_acc,
            'train_time': train_time,
            'history': history,
            'params_efficiency': best_val_acc / (num_params / 1e6)  # 每百萬參數的準確率
        }
        
        self.results[arch_name] = result
        self.save_intermediate_results()
        
        return result
    
    def train_epoch(self, model, loader, criterion, optimizer, scaler):
        """訓練一個epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(loader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(loader), 100. * correct / total
    
    def validate(self, model, loader, criterion):
        """驗證模型"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(loader), 100. * correct / total
    
    def save_intermediate_results(self):
        """保存中間結果"""
        with open(os.path.join(self.save_dir, 'intermediate_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_comparison_report(self):
        """生成比較報告"""
        print("\n生成架構比較報告...")
        
        # 創建性能排名
        rankings = []
        for arch, result in self.results.items():
            rankings.append({
                'Architecture': arch,
                'Best Val Acc (%)': result['best_val_acc'],
                'Parameters (M)': result['num_params'] / 1e6,
                'Efficiency (Acc/M params)': result['params_efficiency'],
                'Train Time (min)': result['train_time'] / 60,
                'Gap to Target (%)': 48.40 - result['best_val_acc']
            })
        
        df = pd.DataFrame(rankings)
        df = df.sort_values('Best Val Acc (%)', ascending=False)
        df.to_csv(os.path.join(self.save_dir, 'performance_ranking.csv'), index=False)
        
        # 生成Markdown報告
        self.generate_markdown_report(df)
        
        # 生成訓練曲線比較圖
        self.plot_training_curves()
        
        # 生成效率分析圖
        self.plot_efficiency_analysis()
        
        print(f"報告已生成到 {self.save_dir} 目錄")
        
        return df
    
    def generate_markdown_report(self, df):
        """生成Markdown格式報告"""
        report = f"""# 架構自動化測試報告

**測試日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**目標準確率**: 48.40% (ResNet34的90%)

## 性能排名

{df.to_markdown(index=False, floatfmt='.2f')}

## 最佳架構分析

### Top 3 架構
"""
        
        for i, row in df.head(3).iterrows():
            arch = row['Architecture']
            result = self.results[arch]
            report += f"""
#### {i+1}. {arch}
- **準確率**: {row['Best Val Acc (%)']:.2f}%
- **參數量**: {row['Parameters (M)']:.2f}M
- **效率**: {row['Efficiency (Acc/M params)']:.2f}
- **距離目標**: {row['Gap to Target (%)']:.2f}%
- **配置**: {json.dumps(result['config'], indent=2)}
"""
        
        # 添加建議
        best_arch = df.iloc[0]['Architecture']
        best_acc = df.iloc[0]['Best Val Acc (%)']
        
        if best_acc >= 48.40:
            status = "✅ **目標達成！**"
            recommendation = f"架構 {best_arch} 已達到目標，建議進行更長時間的訓練以進一步提升性能。"
        elif best_acc >= 45.0:
            status = "⚠️ **接近目標**"
            recommendation = f"架構 {best_arch} 非常接近目標，建議：\n1. 增加訓練epochs到30-50\n2. 使用更強的數據增強\n3. 嘗試集成學習"
        else:
            status = "❌ **需要改進**"
            recommendation = "所有架構都未能接近目標，建議：\n1. 考慮增加模型深度到5-6層\n2. 使用預訓練權重初始化\n3. 重新設計架構"
        
        report += f"""
## 結論與建議

**狀態**: {status}
**最佳架構**: {best_arch} ({best_acc:.2f}%)

### 建議下一步
{recommendation}

### 參數效率分析
- 最高效架構: {df.loc[df['Efficiency (Acc/M params)'].idxmax(), 'Architecture']}
- 最快訓練: {df.loc[df['Train Time (min)'].idxmin(), 'Architecture']}
"""
        
        with open(os.path.join(self.save_dir, 'architecture_comparison_report.md'), 'w') as f:
            f.write(report)
    
    def plot_training_curves(self):
        """繪製訓練曲線比較圖"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 選擇top 5架構繪製
        sorted_archs = sorted(self.results.items(), 
                            key=lambda x: x[1]['best_val_acc'], 
                            reverse=True)[:5]
        
        for arch, result in sorted_archs:
            history = result['history']
            epochs = range(1, len(history['train_acc']) + 1)
            
            # 訓練準確率
            axes[0, 0].plot(epochs, history['train_acc'], 
                          label=f"{arch} ({result['best_val_acc']:.1f}%)", 
                          linewidth=2)
            
            # 驗證準確率
            axes[0, 1].plot(epochs, history['val_acc'], 
                          label=f"{arch} ({result['best_val_acc']:.1f}%)", 
                          linewidth=2)
            
            # 訓練損失
            axes[1, 0].plot(epochs, history['train_loss'], 
                          label=arch, linewidth=2)
            
            # 學習率
            axes[1, 1].plot(epochs, history['lr'], 
                          label=arch, linewidth=2)
        
        # 添加目標線
        axes[0, 1].axhline(y=48.40, color='red', linestyle='--', 
                         label='Target (48.40%)', linewidth=2)
        
        # 設置標題和標籤
        axes[0, 0].set_title('Training Accuracy', fontsize=14)
        axes[0, 1].set_title('Validation Accuracy', fontsize=14)
        axes[1, 0].set_title('Training Loss', fontsize=14)
        axes[1, 1].set_title('Learning Rate', fontsize=14)
        
        for ax in axes.flat:
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_efficiency_analysis(self):
        """繪製效率分析圖"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 準確率 vs 參數量
        for arch, result in self.results.items():
            params_m = result['num_params'] / 1e6
            acc = result['best_val_acc']
            axes[0].scatter(params_m, acc, s=100)
            axes[0].annotate(arch, (params_m, acc), fontsize=8, 
                           xytext=(5, 5), textcoords='offset points')
        
        axes[0].axhline(y=48.40, color='red', linestyle='--', label='Target')
        axes[0].set_xlabel('Parameters (Millions)')
        axes[0].set_ylabel('Validation Accuracy (%)')
        axes[0].set_title('Accuracy vs Model Size')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 訓練時間 vs 準確率
        for arch, result in self.results.items():
            time_min = result['train_time'] / 60
            acc = result['best_val_acc']
            axes[1].scatter(time_min, acc, s=100)
            axes[1].annotate(arch, (time_min, acc), fontsize=8,
                           xytext=(5, 5), textcoords='offset points')
        
        axes[1].axhline(y=48.40, color='red', linestyle='--', label='Target')
        axes[1].set_xlabel('Training Time (minutes)')
        axes[1].set_ylabel('Validation Accuracy (%)')
        axes[1].set_title('Accuracy vs Training Time')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'efficiency_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_all_tests(self, architectures=None, epochs=10):
        """運行所有架構測試"""
        if architectures is None:
            architectures = [
                "wide_convnext", "resnest_4layer", "mini_swin", 
                "efficientnet_style", "wide_resnet", "attention_cnn",
                "convmixer_style", "multiscale_cnn", "dense_efficient", 
                "ghostnet_style"
            ]
        
        print(f"開始測試 {len(architectures)} 個架構...")
        print(f"每個架構訓練 {epochs} epochs")
        print(f"預計總時間: {len(architectures) * 20}-{len(architectures) * 30} 分鐘")
        
        start_time = time.time()
        
        for i, arch in enumerate(architectures):
            print(f"\n進度: {i+1}/{len(architectures)}")
            try:
                self.train_architecture(arch, epochs=epochs)
            except Exception as e:
                print(f"❌ 架構 {arch} 測試失敗: {e}")
                self.results[arch] = {
                    'architecture': arch,
                    'error': str(e),
                    'best_val_acc': 0.0,
                    'num_params': 0,
                    'train_time': 0,
                    'params_efficiency': 0
                }
        
        total_time = time.time() - start_time
        print(f"\n所有測試完成！總時間: {total_time/60:.1f} 分鐘")
        
        # 生成最終報告
        df = self.generate_comparison_report()
        
        # 顯示結果摘要
        print("\n" + "="*60)
        print("測試結果摘要")
        print("="*60)
        print(df.head(10).to_string(index=False))
        
        return self.results


# 以下將實現所有10種架構的類定義...
# (由於篇幅限制，這裡先展示主要框架，具體架構實現將在下一部分)