"""
下载真实鞋子照片数据集 - UT Zappos50K
包含50,025张真实鞋子图片，分为4大类
"""
import os
import urllib.request
import zipfile
from pathlib import Path
import json

print("=" * 70)
print("真实鞋子数据集下载工具")
print("=" * 70)

# 数据集配置
DATASET_URLS = {
    'images': 'http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images-square.zip',
    'metadata': 'http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-data.zip',
}

DATASET_DIR = 'training_data/shoe_real'
os.makedirs(DATASET_DIR, exist_ok=True)

# 类别映射（将UT-Zap50K的4类映射到我们的5类）
CATEGORY_MAPPING = {
    'Shoes': 'low_shoes',      # 普通鞋子映射为低帮鞋
    'Sandals': 'sandals',       # 凉鞋
    'Slippers': 'sandals',      # 拖鞋也归为凉鞋类
    'Boots': 'boots',           # 靴子
}


def download_file(url: str, output_path: str):
    """下载文件并显示进度"""
    print(f"\n下载: {url}")
    print(f"保存到: {output_path}")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\r  进度: {percent}%", end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print("\n  ✓ 下载完成")
        return True
    except Exception as e:
        print(f"\n  ✗ 下载失败: {e}")
        return False


def extract_zip(zip_path: str, extract_dir: str):
    """解压zip文件"""
    print(f"\n解压: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"  ✓ 解压完成")
        return True
    except Exception as e:
        print(f"  ✗ 解压失败: {e}")
        return False


def organize_dataset():
    """整理数据集为训练格式"""
    print("\n[整理数据集]")
    
    # 查找解压后的图片目录
    possible_dirs = [
        f'{DATASET_DIR}/ut-zap50k-images-square',
        f'{DATASET_DIR}/UT-Zap50K',
    ]
    
    image_dir = None
    for d in possible_dirs:
        if os.path.exists(d):
            image_dir = d
            break
    
    if not image_dir:
        print("  ✗ 未找到图片目录")
        return False
    
    print(f"  图片目录: {image_dir}")
    
    # 创建输出目录
    output_images = f'{DATASET_DIR}/images'
    output_labels = f'{DATASET_DIR}/labels'
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)
    
    # 统计
    stats = {'total': 0, 'by_category': {}}
    
    # 遍历类别目录
    for category in ['Shoes', 'Sandals', 'Slippers', 'Boots']:
        category_dir = Path(image_dir) / category
        if not category_dir.exists():
            continue
        
        mapped_class = CATEGORY_MAPPING.get(category, 'unknown')
        class_id = ['sandals', 'low_shoes', 'ankle_shoes', 'boots', 'high_boots'].index(mapped_class) if mapped_class in ['sandals', 'low_shoes', 'ankle_shoes', 'boots', 'high_boots'] else -1
        
        if class_id == -1:
            continue
        
        # 复制图片
        count = 0
        for img_file in category_dir.glob('*.jpg'):
            # 重命名为统一格式
            new_name = f"{mapped_class}_{count:05d}.jpg"
            src = str(img_file)
            dst = f'{output_images}/{new_name}'
            
            # 复制文件
            import shutil
            shutil.copy2(src, dst)
            
            # 创建标注文件（分类任务，整图标注）
            label_file = f'{output_labels}/{new_name.replace(".jpg", ".txt")}'
            with open(label_file, 'w') as f:
                f.write(f"{class_id}\n")
            
            count += 1
        
        stats['by_category'][mapped_class] = count
        stats['total'] += count
        print(f"  {category} -> {mapped_class}: {count} 张")
    
    # 保存类别映射
    mapping_file = f'{DATASET_DIR}/category_mapping.json'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump({
            'source_categories': list(CATEGORY_MAPPING.keys()),
            'target_categories': ['sandals', 'low_shoes', 'ankle_shoes', 'boots', 'high_boots'],
            'mapping': CATEGORY_MAPPING,
            'stats': stats
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n  ✓ 整理完成")
    print(f"  总计: {stats['total']} 张图片")
    print(f"  类别映射: {mapping_file}")
    
    return True


def create_dataset_info():
    """创建数据集信息文件"""
    info = {
        'name': 'UT Zappos50K Real Shoe Dataset',
        'source': 'http://vision.cs.utexas.edu/projects/finegrained/utzap50k/',
        'description': '50,025 real shoe images from Zappos.com',
        'categories': {
            '0': 'sandals',
            '1': 'low_shoes',
            '2': 'ankle_shoes',
            '3': 'boots',
            '4': 'high_boots'
        },
        'usage': 'Academic use only'
    }
    
    info_file = f'{DATASET_DIR}/dataset_info.json'
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"\n数据集信息: {info_file}")


def main():
    """主函数"""
    print("\n UT Zappos50K 数据集")
    print("- 50,025张真实鞋子图片")
    print("- 4大类: Shoes, Sandals, Slippers, Boots")
    print("- 学术使用许可")
    print("-" * 70)
    
    # 询问是否下载
    choice = input("\n是否下载数据集? (y/n): ").strip().lower()
    if choice != 'y':
        print("取消下载")
        return
    
    # 下载图片
    zip_path = f'{DATASET_DIR}/images.zip'
    if not os.path.exists(zip_path):
        if not download_file(DATASET_URLS['images'], zip_path):
            print("\n✗ 下载失败，请手动下载:")
            print(f"  {DATASET_URLS['images']}")
            print(f"  解压到: {DATASET_DIR}")
            return
    else:
        print(f"\n  已存在: {zip_path}")
    
    # 解压
    if not extract_zip(zip_path, DATASET_DIR):
        return
    
    # 整理数据集
    if organize_dataset():
        create_dataset_info()
        
        print("\n" + "=" * 70)
        print("数据集准备完成!")
        print("=" * 70)
        print(f"\n数据位置: {DATASET_DIR}")
        print("\n使用方式:")
        print("  1. 训练模型: python train_shoe_model_real.py")
        print("  2. 数据已按类别整理好，可直接用于训练")
        print("\n注意: 此数据集仅供学术研究使用")
    else:
        print("\n✗ 数据集整理失败")


if __name__ == '__main__':
    main()
