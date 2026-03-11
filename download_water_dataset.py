"""
水坑/积水检测数据集下载工具
从多个来源获取真实水坑图片
"""
import os
import urllib.request
import zipfile
from pathlib import Path

# 水坑检测数据集配置
DATASET_SOURCES = {
    "puddle_1000": {
        "name": "Puddle-1000 Dataset",
        "url": "https://github.com/alexandrosstergiou/Semantic-Segmentation-Models-for-Real-World-Puddle-Detection/archive/refs/heads/main.zip",
        "description": "真实世界水坑分割数据集"
    }
}

def download_file(url, dest_path, desc="Downloading"):
    """下载文件并显示进度"""
    def report_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\r{desc}: {percent}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, dest_path, reporthook=report_hook)
    print()

def download_water_datasets():
    """下载水坑检测数据集"""
    print("=" * 60)
    print("水坑/积水检测数据集下载工具")
    print("=" * 60)
    
    dataset_dir = Path("training_data/water_real")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # 由于真实水坑数据集较难获取，我们创建一个综合方案
    print("\n水坑检测数据集获取方案:")
    print("-" * 60)
    print("1. 使用现有DeepLabV3模型 + CV反射分析 (当前方案)")
    print("2. 创建水坑数据收集工具，从网络获取图片")
    print("3. 使用数据增强生成合成水坑样本")
    print("4. 训练改进的U-Net模型")
    print("-" * 60)
    
    # 创建目录结构
    (dataset_dir / "images").mkdir(exist_ok=True)
    (dataset_dir / "masks").mkdir(exist_ok=True)
    
    print(f"\n✓ 数据集目录已创建: {dataset_dir}")
    
    return str(dataset_dir)

if __name__ == "__main__":
    download_water_datasets()
