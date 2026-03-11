"""
数据收集工具 - 从应用运行时自动收集训练样本
可以手动确认后保存，用于后续模型训练
"""
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import json

class DataCollector:
    """训练数据收集器"""
    
    def __init__(self):
        self.base_dir = 'training_data/collected'
        self.water_dir = f'{self.base_dir}/water'
        self.obstacle_dir = f'{self.base_dir}/obstacle'
        
        # 创建目录
        for d in [self.water_dir, self.obstacle_dir]:
            os.makedirs(f'{d}/images', exist_ok=True)
            os.makedirs(f'{d}/annotations', exist_ok=True)
        
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.collected_count = {'water': 0, 'obstacle': 0}
        
        print(f"[数据收集器] 会话ID: {self.session_id}")
        print(f"  保存路径: {self.base_dir}")
    
    def save_water_sample(self, image: np.ndarray, mask: np.ndarray, 
                          confidence: float, auto_save: bool = False):
        """
        保存水坑检测样本
        
        Args:
            image: 原始图像 (BGR格式)
            mask: 分割掩码 (二值图像)
            confidence: 检测置信度
            auto_save: 是否自动保存（否则只保存高置信度样本）
        """
        # 只保存高置信度样本，或手动确认
        if not auto_save and confidence < 0.8:
            return False
        
        timestamp = datetime.now().strftime('%H%M%S_%f')
        filename = f'water_{self.session_id}_{timestamp}'
        
        # 保存原图
        img_path = f'{self.water_dir}/images/{filename}.jpg'
        cv2.imwrite(img_path, image)
        
        # 保存掩码
        mask_path = f'{self.water_dir}/annotations/{filename}.png'
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        
        # 保存元数据
        meta = {
            'timestamp': timestamp,
            'confidence': float(confidence),
            'image_shape': image.shape[:2],
            'mask_pixels': int(np.sum(mask))
        }
        meta_path = f'{self.water_dir}/annotations/{filename}.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        self.collected_count['water'] += 1
        print(f"[保存水坑样本] {filename} (置信度: {confidence:.2f})")
        return True
    
    def save_obstacle_sample(self, image: np.ndarray, 
                            detections: list, auto_save: bool = False):
        """
        保存障碍物检测样本
        
        Args:
            image: 原始图像
            detections: 检测结果列表 [(class_id, x, y, w, h, conf), ...]
            auto_save: 是否自动保存
        """
        if not detections:
            return False
        
        # 只保存有高置信度检测的样本
        max_conf = max([d[5] for d in detections]) if detections else 0
        if not auto_save and max_conf < 0.7:
            return False
        
        timestamp = datetime.now().strftime('%H%M%S_%f')
        filename = f'obstacle_{self.session_id}_{timestamp}'
        
        # 保存原图
        img_path = f'{self.obstacle_dir}/images/{filename}.jpg'
        cv2.imwrite(img_path, image)
        
        # 保存YOLO格式标注
        label_path = f'{self.obstacle_dir}/annotations/{filename}.txt'
        h, w = image.shape[:2]
        
        with open(label_path, 'w') as f:
            for det in detections:
                class_id, x, y, bw, bh, conf = det
                # 转换为YOLO格式（归一化）
                x_center = x / w
                y_center = y / h
                width = bw / w
                height = bh / h
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # 保存元数据
        meta = {
            'timestamp': timestamp,
            'num_detections': len(detections),
            'max_confidence': float(max_conf),
            'detections': [
                {'class': int(d[0]), 'conf': float(d[5])} for d in detections
            ]
        }
        meta_path = f'{self.obstacle_dir}/annotations/{filename}.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        self.collected_count['obstacle'] += 1
        print(f"[保存障碍物样本] {filename} (检测到 {len(detections)} 个目标)")
        return True
    
    def get_stats(self):
        """获取收集统计"""
        return {
            'session_id': self.session_id,
            'water_samples': self.collected_count['water'],
            'obstacle_samples': self.collected_count['obstacle'],
            'total': sum(self.collected_count.values())
        }
    
    def export_to_training_set(self, min_confidence: float = 0.8):
        """
        将收集的数据导出到正式训练集
        只导出高质量样本
        """
        print("\n[导出训练集]")
        
        exported = {'water': 0, 'obstacle': 0}
        
        # 导出水坑样本
        water_annotations = Path(self.water_dir) / 'annotations'
        for meta_file in water_annotations.glob('*.json'):
            with open(meta_file) as f:
                meta = json.load(f)
            
            if meta['confidence'] >= min_confidence:
                base_name = meta_file.stem
                # 复制到正式训练目录
                src_img = f'{self.water_dir}/images/{base_name}.jpg'
                src_mask = f'{self.water_dir}/annotations/{base_name}.png'
                
                dst_img = f'training_data/water/images/{base_name}.jpg'
                dst_mask = f'training_data/water/masks/{base_name}.png'
                
                if os.path.exists(src_img):
                    import shutil
                    shutil.copy2(src_img, dst_img)
                    shutil.copy2(src_mask, dst_mask)
                    exported['water'] += 1
        
        # 导出障碍物样本
        obstacle_annotations = Path(self.obstacle_dir) / 'annotations'
        for meta_file in obstacle_annotations.glob('*.json'):
            with open(meta_file) as f:
                meta = json.load(f)
            
            if meta['max_confidence'] >= min_confidence:
                base_name = meta_file.stem
                src_img = f'{self.obstacle_dir}/images/{base_name}.jpg'
                src_label = f'{self.obstacle_dir}/annotations/{base_name}.txt'
                
                dst_img = f'training_data/obstacle/images/{base_name}.jpg'
                dst_label = f'training_data/obstacle/labels/{base_name}.txt'
                
                if os.path.exists(src_img):
                    import shutil
                    shutil.copy2(src_img, dst_img)
                    shutil.copy2(src_label, dst_label)
                    exported['obstacle'] += 1
        
        print(f"  导出水坑样本: {exported['water']}")
        print(f"  导出障碍物样本: {exported['obstacle']}")
        print(f"  总计: {sum(exported.values())}")
        
        return exported


# 全局收集器实例
collector = None

def get_collector():
    """获取数据收集器实例"""
    global collector
    if collector is None:
        collector = DataCollector()
    return collector


if __name__ == '__main__':
    # 测试数据收集器
    print("=" * 60)
    print("数据收集工具测试")
    print("=" * 60)
    
    c = get_collector()
    
    # 模拟保存样本
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_mask = np.random.random((480, 640)) > 0.7
    
    c.save_water_sample(test_img, test_mask.astype(np.float32), 0.85)
    
    print("\n统计:", c.get_stats())
