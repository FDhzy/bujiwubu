"""
整理UT Zappos50K数据集
"""
import os
import shutil
from pathlib import Path
import json

DATASET_DIR = 'training_data/shoe_real'
image_dir = f'{DATASET_DIR}/ut-zap50k-images-square'

# 类别映射
CATEGORY_MAPPING = {
    'Shoes': 'low_shoes',
    'Sandals': 'sandals',
    'Slippers': 'sandals',
    'Boots': 'boots',
}

CLASS_TO_IDX = {
    'sandals': 0,
    'low_shoes': 1,
    'ankle_shoes': 2,
    'boots': 3,
    'high_boots': 4
}

def organize_dataset():
    """整理数据集"""
    # 创建输出目录
    output_images = f'{DATASET_DIR}/images'
    output_labels = f'{DATASET_DIR}/labels'
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)
    
    print('整理数据集...')
    stats = {'total': 0, 'by_category': {}}
    
    for category in ['Shoes', 'Sandals', 'Slippers', 'Boots']:
        category_dir = Path(image_dir) / category
        if not category_dir.exists():
            print(f'  跳过: {category} (目录不存在)')
            continue
        
        mapped_class = CATEGORY_MAPPING.get(category, 'unknown')
        class_id = CLASS_TO_IDX.get(mapped_class, -1)
        if class_id == -1:
            continue
        
        count = 0
        # 递归搜索所有子目录中的jpg文件
        for img_file in category_dir.rglob('*.jpg'):
            new_name = f'{mapped_class}_{count:05d}.jpg'
            src = str(img_file)
            dst = os.path.join(output_images, new_name)
            shutil.copy2(src, dst)
            
            # 创建标注文件
            label_file = os.path.join(output_labels, new_name.replace('.jpg', '.txt'))
            with open(label_file, 'w') as f:
                f.write(f'{class_id}\n')
            
            count += 1
            if count >= 2000:  # 每类最多2000张
                break
        
        stats['by_category'][mapped_class] = count
        stats['total'] += count
        print(f'  {category} -> {mapped_class}: {count} 张')
    
    print(f'\n✓ 整理完成: {stats["total"]} 张图片')
    
    # 保存统计
    with open(f'{DATASET_DIR}/stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

if __name__ == '__main__':
    organize_dataset()
