"""
下载 COCO 数据集并转换为 YOLO 格式用于障碍物检测训练
只下载与盲人导航相关的类别
"""
import os
import sys
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# COCO 与项目类别映射
# COCO 类别ID -> 项目类别ID
COCO_TO_PROJECT_MAPPING = {
    1: 0,   # person -> person
    2: 1,   # bicycle -> bicycle
    3: 2,   # car -> car
    4: 2,   # motorcycle -> motorcycle
    6: 3,   # bus -> car (合并到car)
    8: 3,   # truck -> car (合并到car)
    13: 6,  # bench -> bench
}

# 需要的 COCO 类别
COCO_CLASSES_NEEDED = list(COCO_TO_PROJECT_MAPPING.keys())

# 数据集配置
COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

DATASET_DIR = Path("training_data/coco_for_obstacle")


def download_file(url: str, dest_path: Path, desc: str):
    """下载文件并显示进度"""
    print(f"\n正在下载: {desc}")
    print(f"URL: {url}")
    print(f"保存到: {dest_path}")
    
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\r进度: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print("\n✓ 下载完成!")
        return True
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path):
    """解压 zip 文件"""
    print(f"\n解压: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ 解压完成: {extract_to}")
        return True
    except Exception as e:
        print(f"✗ 解压失败: {e}")
        return False


def convert_coco_to_yolo(annotation_file: Path, image_dir: Path, output_dir: Path, split: str):
    """
    将 COCO 标注转换为 YOLO 格式
    
    Args:
        annotation_file: COCO 标注文件路径
        image_dir: 图片目录
        output_dir: YOLO 格式输出目录
        split: 'train' 或 'val'
    """
    print(f"\n转换 COCO 到 YOLO 格式 ({split})...")
    
    # 创建输出目录
    yolo_images_dir = output_dir / "images" / split
    yolo_labels_dir = output_dir / "labels" / split
    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 COCO 标注
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # 构建类别映射
    coco_categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 构建图片信息映射
    images_info = {img['id']: img for img in coco_data['images']}
    
    # 按图片ID分组标注
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        # 只保留需要的类别
        if ann['category_id'] not in COCO_CLASSES_NEEDED:
            continue
        
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    print(f"找到 {len(annotations_by_image)} 张包含目标类别的图片")
    
    # 转换每张图片的标注
    converted_count = 0
    skipped_count = 0
    
    for img_id, annotations in annotations_by_image.items():
        img_info = images_info[img_id]
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # 源图片路径
        src_img_path = image_dir / img_filename
        if not src_img_path.exists():
            skipped_count += 1
            continue
        
        # 复制图片到 YOLO 目录
        dst_img_path = yolo_images_dir / img_filename
        if not dst_img_path.exists():
            import shutil
            shutil.copy2(src_img_path, dst_img_path)
        
        # 生成 YOLO 格式标注
        label_filename = Path(img_filename).stem + ".txt"
        label_path = yolo_labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # 获取项目类别ID
                project_class_id = COCO_TO_PROJECT_MAPPING.get(ann['category_id'])
                if project_class_id is None:
                    continue
                
                # COCO bbox: [x, y, width, height]
                x, y, w, h = ann['bbox']
                
                # 转换为 YOLO 格式（归一化中心坐标）
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                
                # 确保值在 [0, 1] 范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                f.write(f"{project_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        converted_count += 1
        
        if converted_count % 1000 == 0:
            print(f"  已处理 {converted_count} 张图片...")
    
    print(f"✓ 转换完成: {converted_count} 张图片, 跳过 {skipped_count} 张")
    return converted_count


def create_dataset_config(output_dir: Path, num_samples: int):
    """创建数据集配置文件"""
    config = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": 7,  # 7个类别
        "names": {
            0: "person",
            1: "bicycle",
            2: "motorcycle",
            3: "car",
            4: "pole",
            5: "bench",
            6: "trash_can"
        }
    }
    
    config_path = output_dir / "dataset.yaml"
    import yaml
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    
    print(f"\n✓ 数据集配置已保存: {config_path}")


def main():
    """主函数"""
    print("=" * 70)
    print("下载 COCO 数据集并转换为 YOLO 格式")
    print("=" * 70)
    print("\n将下载以下类别:")
    for coco_id, proj_id in COCO_TO_PROJECT_MAPPING.items():
        print(f"  COCO {coco_id} -> Project {proj_id}")
    
    print("\n数据量:")
    print("  - 验证集: 约 1GB (5,000 张图片)")
    print("  - 训练集: 约 18GB (118,000 张图片)")
    print("\n建议: 先下载验证集进行测试，确认无误后再下载训练集")
    print("-" * 70)
    
    # 创建临时下载目录
    temp_dir = DATASET_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 下载标注文件
        ann_zip = temp_dir / "annotations_trainval2017.zip"
        ann_dir = temp_dir / "annotations"
        
        if not ann_zip.exists():
            print("\n[步骤 1/3] 下载标注文件 (~250MB)")
            if not download_file(COCO_URLS["annotations"], ann_zip, "COCO 标注文件"):
                print("下载失败，退出")
                return
        else:
            print(f"\n标注文件已存在: {ann_zip}")
        
        # 解压标注文件（如果还没解压）
        if not ann_dir.exists() or not (ann_dir / "instances_val2017.json").exists():
            print("\n解压标注文件...")
            extract_zip(ann_zip, temp_dir)
        else:
            print("标注文件已解压")
        
        # 先下载验证集（较小，用于测试）
        print("\n[步骤 2/3] 下载验证集")
        download_val = input("是否下载验证集? (约 1GB, 推荐先下载这个) (y/n): ").strip().lower() == 'y'
        
        if download_val:
            val_zip = temp_dir / "val2017.zip"
            if not val_zip.exists():
                if not download_file(COCO_URLS["val_images"], val_zip, "COCO 验证集图片"):
                    print("下载失败")
                    return
                extract_zip(val_zip, temp_dir)
            else:
                print(f"\n验证集已存在: {val_zip}")
            
            # 转换验证集
            val_ann = temp_dir / "annotations" / "instances_val2017.json"
            val_img_dir = temp_dir / "val2017"
            val_count = convert_coco_to_yolo(val_ann, val_img_dir, DATASET_DIR, "val")
        
        # 询问是否下载训练集
        print("\n[步骤 3/3] 下载训练集")
        download_train = input("是否下载训练集? (约 18GB, 耗时较长) (y/n): ").strip().lower() == 'y'
        
        if download_train:
            train_zip = temp_dir / "train2017.zip"
            if not train_zip.exists():
                if not download_file(COCO_URLS["train_images"], train_zip, "COCO 训练集图片"):
                    print("下载失败")
                    return
                extract_zip(train_zip, temp_dir)
            else:
                print(f"\n训练集已存在: {train_zip}")
            
            # 转换训练集
            train_ann = temp_dir / "annotations" / "instances_train2017.json"
            train_img_dir = temp_dir / "train2017"
            train_count = convert_coco_to_yolo(train_ann, train_img_dir, DATASET_DIR, "train")
        
        # 创建配置文件
        create_dataset_config(DATASET_DIR, 0)
        
        print("\n" + "=" * 70)
        print("COCO 数据集准备完成!")
        print("=" * 70)
        print(f"\n数据集位置: {DATASET_DIR}")
        print("\n下一步:")
        print("  1. 运行: python obstacle_data_collector.py")
        print("  2. 选择 '导入 COCO 数据集'")
        print("  3. 输入路径导入到训练目录")
        print("  4. 使用摄像头补充特定类别（如 stairs, wire 等）")
        print("  5. 运行: python train_obstacle_model_custom.py")
        
    except KeyboardInterrupt:
        print("\n\n用户取消下载")
    except Exception as e:
        print(f"\n错误: {e}")


if __name__ == "__main__":
    main()
