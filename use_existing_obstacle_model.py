"""
使用已有的障碍物检测模型
将 training_checkpoints/obstacle/best_model.pth 转换为 TFLite 格式
"""
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '.')

MODEL_PATH = Path('training_checkpoints/obstacle/best_model.pth')
OUTPUT_PATH = Path('assets/models/yolov8_obstacle.tflite')


def inspect_model():
    """检查模型结构"""
    print("=" * 70)
    print("检查现有障碍物模型")
    print("=" * 70)
    
    if not MODEL_PATH.exists():
        print(f"✗ 模型不存在: {MODEL_PATH}")
        return False
    
    print(f"\n模型文件: {MODEL_PATH}")
    print(f"大小: {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    
    try:
        # 加载模型
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        print("\n模型内容:")
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                print(f"  - {key}")
            
            # 检查是否有模型状态
            if 'model' in checkpoint:
                print("\n✓ 找到模型状态字典")
            elif 'state_dict' in checkpoint:
                print("\n✓ 找到状态字典")
            else:
                print("\n⚠ 未找到标准状态字典，可能是完整模型")
        else:
            print("\n✓ 加载的是完整模型对象")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 加载失败: {e}")
        return False


def convert_to_tflite():
    """转换为 TFLite 格式"""
    print("\n" + "=" * 70)
    print("转换为 TFLite 格式")
    print("=" * 70)
    
    try:
        # 这里需要根据实际模型架构进行转换
        # 由于我们不知道确切架构，先尝试直接复制为备用方案
        
        print("\n注意: 直接转换 PyTorch 模型到 TFLite 需要知道模型架构")
        print("建议方案:")
        print("  1. 使用 PyTorch 模型直接进行推理")
        print("  2. 或者重新训练并导出为 TFLite")
        
        # 创建一个标记文件，表示使用 PyTorch 模型
        marker_file = Path('assets/models/use_pytorch_obstacle.txt')
        with open(marker_file, 'w') as f:
            f.write(str(MODEL_PATH.absolute()))
        
        print(f"\n✓ 已创建标记文件: {marker_file}")
        print(f"  模型路径: {MODEL_PATH}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 转换失败: {e}")
        return False


def main():
    """主函数"""
    if inspect_model():
        print("\n" + "=" * 70)
        print("模型检查完成!")
        print("=" * 70)
        print("\n你有两个选择:")
        print("  1. 使用 PyTorch 模型直接推理（需要修改代码）")
        print("  2. 使用 CV 备用检测器（立即可用）")
        
        choice = input("\n选择 (1-2, 默认2): ").strip() or "2"
        
        if choice == "1":
            convert_to_tflite()
            print("\n请修改 obstacle_detector.py 以支持 PyTorch 模型")
        else:
            print("\n将使用 CV 备用检测器")
            print("如需启用 ML 检测，请解决模型加载问题")
    else:
        print("\n无法使用现有模型")


if __name__ == "__main__":
    main()
