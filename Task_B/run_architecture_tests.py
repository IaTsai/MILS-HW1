#!/usr/bin/env python3
"""
啟動自動化架構測試
預計運行時間: 3-5小時
"""

import os
import sys
import argparse
from auto_test_architectures import ArchitectureTester


def main():
    parser = argparse.ArgumentParser(description='自動化架構測試系統')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='每個架構訓練的epochs數 (預設: 10)')
    parser.add_argument('--architectures', nargs='+', 
                       help='要測試的架構列表 (預設: 全部)')
    parser.add_argument('--skip', nargs='+', 
                       help='要跳過的架構列表')
    parser.add_argument('--save_dir', type=str, default='architecture_results',
                       help='保存結果的目錄')
    args = parser.parse_args()
    
    # 所有可用架構
    all_architectures = [
        "wide_convnext", "resnest_4layer", "mini_swin", 
        "efficientnet_style", "wide_resnet", "attention_cnn",
        "convmixer_style", "multiscale_cnn", "dense_efficient", 
        "ghostnet_style"
    ]
    
    # 決定要測試的架構
    if args.architectures:
        architectures = args.architectures
    else:
        architectures = all_architectures
    
    # 排除要跳過的架構
    if args.skip:
        architectures = [arch for arch in architectures if arch not in args.skip]
    
    print("="*60)
    print("自動化架構測試系統")
    print("="*60)
    print(f"測試架構數量: {len(architectures)}")
    print(f"每個架構訓練: {args.epochs} epochs")
    print(f"結果保存到: {args.save_dir}")
    print(f"架構列表: {', '.join(architectures)}")
    print("="*60)
    
    # 創建測試器並運行
    tester = ArchitectureTester(save_dir=args.save_dir)
    
    try:
        results = tester.run_all_tests(architectures=architectures, epochs=args.epochs)
        
        # 顯示最終結果摘要
        print("\n" + "="*60)
        print("測試完成！最終結果摘要:")
        print("="*60)
        
        # 找出最佳架構
        best_arch = max(results.items(), key=lambda x: x[1].get('best_val_acc', 0))
        print(f"\n🏆 最佳架構: {best_arch[0]}")
        print(f"   準確率: {best_arch[1]['best_val_acc']:.2f}%")
        print(f"   參數量: {best_arch[1]['num_params']/1e6:.2f}M")
        print(f"   目標差距: {48.40 - best_arch[1]['best_val_acc']:.2f}%")
        
        # 檢查是否達到目標
        if best_arch[1]['best_val_acc'] >= 48.40:
            print("\n✅ 恭喜！已達到目標準確率 48.40%!")
        elif best_arch[1]['best_val_acc'] >= 45.0:
            print("\n⚠️ 接近目標！建議進行更長時間的訓練。")
        else:
            print("\n❌ 尚未達到目標，可能需要調整架構或訓練策略。")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 測試被用戶中斷")
        print("中間結果已保存")
    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n詳細報告請查看: {args.save_dir}/architecture_comparison_report.md")


if __name__ == "__main__":
    main()