"""
下载真实水坑图片用于训练
使用多个公开数据集源
"""
import os
import urllib.request
import zipfile
from pathlib import Path
import shutil


def download_puddle_dataset():
    """下载水坑数据集"""
    print("=" * 60)
    print("下载真实水坑数据集")
    print("=" * 60)
    
    dataset_dir = Path("training_data/water_real")
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    
    # 创建目录
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n数据集源:")
    print("-" * 60)
    print("1. Puddle-1000 数据集 (GitHub)")
    print("   - 真实场景水坑图片")
    print("   - 包含分割标注")
    print("\n2. 自建数据集 (推荐)")
    print("   - 使用手机拍摄真实水坑")
    print("   - 使用 water_data_collector.py 标注")
    print("-" * 60)
    
    # 尝试下载 Puddle-1000
    print("\n尝试下载 Puddle-1000 数据集...")
    url = "https://github.com/alexandrosstergiou/Semantic-Segmentation-Models-for-Real-World-Puddle-Detection/archive/refs/heads/main.zip"
    zip_path = dataset_dir / "puddle_dataset.zip"
    
    try:
        print(f"下载中... {url}")
        urllib.request.urlretrieve(url, zip_path)
        print("✓ 下载完成")
        
        # 解压
        print("解压中...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("✓ 解压完成")
        
        # 移动图片到正确位置
        extracted_dir = dataset_dir / "Semantic-Segmentation-Models-for-Real-World-Puddle-Detection-main"
        if extracted_dir.exists():
            # 查找图片目录
            for img_dir in extracted_dir.rglob("images"):
                for img_file in img_dir.glob("*.jpg"):
                    shutil.copy(img_file, image_dir)
            for mask_dir_src in extracted_dir.rglob("masks"):
                for mask_file in mask_dir_src.glob("*.png"):
                    shutil.copy(mask_file, mask_dir)
            
            # 清理
            shutil.rmtree(extracted_dir)
            zip_path.unlink()
            
        print("✓ 数据集整理完成")
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("\n建议: 使用合成数据或手动收集图片")
    
    # 统计
    image_count = len(list(image_dir.glob("*.jpg")))
    mask_count = len(list(mask_dir.glob("*.png")))
    
    print(f"\n数据集统计:")
    print(f"  图片: {image_count} 张")
    print(f"  标注: {mask_count} 张")
    print(f"  位置: {dataset_dir}")
    
    return image_count


if __name__ == "__main__":
    download_puddle_dataset()
