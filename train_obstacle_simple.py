"""
简化版障碍物检测训练 - 使用Ultralytics YOLOv8
需要先安装: pip install ultralytics
"""
import os
from pathlib import Path
import yaml

# 配置
DATASET_DIR = Path('training_data/obstacle_custom')
OUTPUT_DIR = Path('training_checkpoints/obstacle')

# 创建YOLO配置文件
def create_yolo_config():
    """创建YOLOv8数据集配置"""
    config = {
        'path': str(DATASET_DIR.absolute()),
        'train': 'images',
        'val': 'images',  # 简单起见，使用相同数据
        'names': {
            0: 'person',
            1: 'bicycle',
            2: 'motorcycle'
        }
    }
    
    config_path = DATASET_DIR / 'yolo_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    
    return config_path


def train_with_ultralytics(epochs=30, batch_size=8):
    """使用Ultralytics YOLOv8训练"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("请先安装 ultralytics:")
        print("  pip install ultralytics")
        return False
    
    print("=" * 70)
    print("使用 Ultralytics YOLOv8 训练障碍物检测模型")
    print("=" * 70)
    
    # 创建配置
    config_path = create_yolo_config()
    print(f"\n数据集配置: {config_path}")
    
    # 加载预训练模型
    print("\n加载 YOLOv8n 预训练模型...")
    model = YOLO('yolov8n.pt')
    
    # 训练
    print(f"\n开始训练:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  数据集: {DATASET_DIR}")
    
    results = model.train(
        data=str(config_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=256,
        project=str(OUTPUT_DIR),
        name='obstacle_model',
        exist_ok=True,
        device='cpu'
    )
    
    # 导出为TFLite
    print("\n导出为 TFLite 格式...")
    model.export(format='tflite', imgsz=256)
    
    # 复制到assets目录
    import shutil
    tflite_path = OUTPUT_DIR / 'obstacle_model' / 'weights' / 'best.tflite'
    if tflite_path.exists():
        dest_path = Path('assets/models/yolov8_obstacle.tflite')
        shutil.copy2(tflite_path, dest_path)
        print(f"✓ 模型已保存到: {dest_path}")
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    return True


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    
    # 检查数据
    images_dir = DATASET_DIR / 'images'
    if not images_dir.exists() or len(list(images_dir.glob('*.jpg'))) == 0:
        print("错误: 没有找到训练数据")
        print("请先运行: python download_roboflow_obstacle.py")
        print("选择选项1生成合成数据")
        return
    
    # 训练
    success = train_with_ultralytics(args.epochs, args.batch_size)
    
    if not success:
        print("\n备用方案: 使用 PyTorch 训练")
        print("运行: python train_obstacle_model_custom.py")


if __name__ == "__main__":
    main()
