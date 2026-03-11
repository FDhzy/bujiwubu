"""
从 Roboflow 下载公开障碍物数据集
比 COCO 更小、更针对障碍物检测场景
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Roboflow 公开数据集
ROBOFLOW_DATASETS = {
    "street-obstacles": {
        "url": "https://public.roboflow.com/ds/7f1jW8XqQV?key=YOUR_KEY",
        "description": "街道障碍物检测",
        "classes": ["person", "car", "bicycle", "motorcycle", "pole", "trash_can"],
        "size_mb": 150
    }
}

# 备用：使用 Hugging Face 数据集
HUGGINGFACE_DATASETS = {
    "coco-person-car": {
        "url": "https://huggingface.co/datasets/huggingface/coco/resolve/main/coco-2017.zip",
        "description": "COCO 精简版（只含 person, car, bicycle）",
        "size_mb": 500
    }
}

DATASET_DIR = Path("training_data/obstacle_custom")


def download_with_progress(url: str, dest_path: Path, desc: str):
    """下载文件并显示进度"""
    print(f"\n正在下载: {desc}")
    print(f"保存到: {dest_path}")
    
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            mb_downloaded = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            sys.stdout.write(f"\r进度: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print("\n✓ 下载完成!")
        return True
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False


def extract_and_organize(zip_path: Path, extract_to: Path):
    """解压并整理数据集"""
    print(f"\n解压: {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ 解压完成")
        
        # 查找解压后的目录结构
        extracted_dirs = [d for d in extract_to.iterdir() if d.is_dir()]
        
        for data_dir in extracted_dirs:
            # 查找 train/test/valid 目录
            for split in ['train', 'test', 'valid']:
                split_dir = data_dir / split
                if split_dir.exists():
                    images_dir = split_dir / 'images'
                    labels_dir = split_dir / 'labels'
                    
                    if images_dir.exists():
                        print(f"  找到 {split} 数据: {len(list(images_dir.glob('*')))} 张图片")
        
        return True
        
    except Exception as e:
        print(f"✗ 解压失败: {e}")
        return False


def generate_synthetic_obstacle_data(num_samples: int = 100):
    """
    生成合成障碍物数据用于快速测试
    当无法下载真实数据时使用
    """
    import cv2
    import numpy as np
    from datetime import datetime
    
    print(f"\n生成合成障碍物数据 ({num_samples} 张)...")
    
    images_dir = DATASET_DIR / 'images'
    labels_dir = DATASET_DIR / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 障碍物类别定义
    obstacle_types = [
        {'name': 'person', 'aspect_range': (2.0, 4.0), 'color_range': ((50, 50, 50), (150, 150, 150))},
        {'name': 'pole', 'aspect_range': (5.0, 10.0), 'color_range': ((60, 60, 60), (100, 100, 100))},
        {'name': 'car', 'aspect_range': (0.5, 2.0), 'color_range': ((40, 40, 40), (120, 120, 120))},
    ]
    
    for i in range(num_samples):
        # 创建背景
        img = np.ones((480, 640, 3), dtype=np.uint8) * np.random.randint(100, 200)
        
        # 添加随机噪声模拟地面纹理
        noise = np.random.randint(-20, 20, (480, 640, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        annotations = []
        
        # 随机添加 1-3 个障碍物
        num_obstacles = np.random.randint(1, 4)
        for _ in range(num_obstacles):
            obs_type = np.random.choice(obstacle_types)
            
            # 随机大小和位置
            w = np.random.randint(30, 150)
            aspect = np.random.uniform(*obs_type['aspect_range'])
            h = int(w * aspect)
            
            x = np.random.randint(50, max(51, 590 - w))
            y = np.random.randint(50, max(51, 400 - h))
            
            # 绘制障碍物
            low, high = obs_type['color_range']
            color = tuple(np.random.randint(low[i], high[i]) for i in range(3))
            cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
            
            # 添加阴影效果
            shadow_color = tuple(max(0, c - 30) for c in color)
            cv2.rectangle(img, (x+5, y+h-10), (x+w, y+h), shadow_color, -1)
            
            # 记录标注
            class_id = obstacle_types.index(obs_type)
            x_center = (x + w/2) / 640
            y_center = (y + h/2) / 480
            width = w / 640
            height = h / 480
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # 保存图片
        img_path = images_dir / f'synthetic_{i:04d}.jpg'
        cv2.imwrite(str(img_path), img)
        
        # 保存标注
        label_path = labels_dir / f'synthetic_{i:04d}.txt'
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        if (i + 1) % 20 == 0:
            print(f"  已生成 {i+1}/{num_samples} 张...")
    
    # 创建类别文件
    with open(DATASET_DIR / 'classes.txt', 'w') as f:
        for obs in obstacle_types:
            f.write(f"{obs['name']}\n")
    
    print(f"✓ 合成数据生成完成: {num_samples} 张图片")
    return num_samples


def main():
    """主函数"""
    print("=" * 70)
    print("障碍物数据集下载工具")
    print("=" * 70)
    
    print("\n选择数据源:")
    print("  1. 生成合成数据 (快速, 用于测试)")
    print("  2. 下载 COCO 数据 (推荐, 数据量大)")
    print("  3. 使用已有图片")
    
    choice = input("\n输入选项 (1-3): ").strip()
    
    if choice == '1':
        num = int(input("生成多少张合成图片? (默认100): ") or "100")
        count = generate_synthetic_obstacle_data(num)
        
        print("\n" + "=" * 70)
        print(f"✓ 已生成 {count} 张合成障碍物图片")
        print("=" * 70)
        print("\n下一步:")
        print("  python train_obstacle_model_custom.py")
        
    elif choice == '2':
        print("\n启动 COCO 数据下载...")
        print("运行: python download_coco_for_obstacle.py")
        os.system("python download_coco_for_obstacle.py")
        
    else:
        print("\n请使用数据收集工具导入已有图片:")
        print("  python obstacle_data_collector.py")


if __name__ == "__main__":
    main()
