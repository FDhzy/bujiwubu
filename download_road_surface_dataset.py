"""
下载 Road Surface Detection 数据集并提取水坑图片
数据集包含水坑(water puddle)标注，可用于增强训练
"""
import os
import sys
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

sys.path.insert(0, '.')

# 数据集下载链接（Hugging Face）
DATASET_URL = "https://huggingface.co/datasets/Shrinidhi/Road-Surface-Detection/resolve/main/dataset.zip"

def download_dataset():
    """下载数据集（带进度显示，支持断点续传）"""
    import urllib.request
    import time
    
    dataset_dir = Path("training_data/water_real")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = dataset_dir / "road_surface_dataset.zip"
    
    if zip_path.exists():
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"数据集已存在: {zip_path} ({size_mb:.1f} MB)")
        return zip_path
    
    print(f"正在下载 Road Surface Detection 数据集...")
    print(f"URL: {DATASET_URL}")
    print(f"保存到: {zip_path}")
    
    try:
        # 带进度回调的下载
        last_print = [0]
        start_time = [time.time()]
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            elapsed = time.time() - start_time[0]
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                mb_down = downloaded / 1024 / 1024
                mb_total = total_size / 1024 / 1024
                speed = mb_down / elapsed if elapsed > 0 else 0
                if pct - last_print[0] >= 5 or pct >= 100:
                    print(f"  进度: {pct:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB) 速度: {speed:.1f} MB/s")
                    last_print[0] = pct
            else:
                mb_down = downloaded / 1024 / 1024
                if mb_down - last_print[0] >= 10:
                    print(f"  已下载: {mb_down:.1f} MB")
                    last_print[0] = mb_down
        
        urllib.request.urlretrieve(DATASET_URL, zip_path, reporthook=progress_hook)
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"✓ 下载完成: {zip_path} ({size_mb:.1f} MB)")
        return zip_path
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        # 清理不完整的文件
        if zip_path.exists():
            zip_path.unlink()
        return None

def extract_water_puddles(zip_path):
    """解压并提取水坑图片和掩码"""
    dataset_dir = Path("training_data/water_real")
    extract_dir = dataset_dir / "road_surface_temp"
    
    print(f"\n解压数据集...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"✓ 解压完成")
    
    # 查找图片和掩码目录
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    
    # 水坑在掩码中的颜色 (Dark Red)
    PUDDLE_COLOR = np.array([128, 0, 0])  # BGR格式
    
    print(f"\n提取水坑图片...")
    puddle_count = 0
    
    # 遍历解压后的目录
    for mask_path in extract_dir.rglob("*mask*.png"):
        # 读取掩码
        mask = cv2.imread(str(mask_path))
        if mask is None:
            continue
        
        # 检查是否有水坑（暗红色区域）
        puddle_mask = cv2.inRange(mask, PUDDLE_COLOR - 10, PUDDLE_COLOR + 10)
        
        if np.sum(puddle_mask) > 1000:  # 有水坑区域
            # 找到对应的图片
            img_path = mask_path.parent / mask_path.name.replace("_mask", "").replace("mask_", "")
            if not img_path.exists():
                # 尝试其他命名模式
                img_path = mask_path.parent / (mask_path.stem.replace("_mask", "").replace("mask_", "") + ".jpg")
            
            if img_path.exists():
                # 复制到训练目录
                new_name = f"road_surface_puddle_{puddle_count:04d}"
                
                # 复制图片
                img = cv2.imread(str(img_path))
                cv2.imwrite(str(image_dir / f"{new_name}.jpg"), img)
                
                # 保存二值掩码
                cv2.imwrite(str(mask_dir / f"{new_name}.png"), puddle_mask)
                
                puddle_count += 1
                
                if puddle_count % 10 == 0:
                    print(f"  已提取 {puddle_count} 张水坑图片...")
    
    print(f"✓ 共提取 {puddle_count} 张水坑图片")
    
    # 清理临时文件
    import shutil
    shutil.rmtree(extract_dir, ignore_errors=True)
    
    return puddle_count

def main():
    print("=" * 60)
    print("下载 Road Surface Detection 数据集")
    print("=" * 60)
    
    # 下载
    zip_path = download_dataset()
    if zip_path is None:
        print("\n下载失败，请手动下载:")
        print(f"  1. 访问: https://huggingface.co/datasets/Shrinidhi/Road-Surface-Detection")
        print(f"  2. 下载 dataset.zip")
        print(f"  3. 放到 training_data/water_real/ 目录")
        return
    
    # 提取水坑
    count = extract_water_puddles(zip_path)
    
    print(f"\n" + "=" * 60)
    print(f"完成！新增 {count} 张水坑训练图片")
    print(f"=" * 60)

if __name__ == "__main__":
    main()
