"""
障碍物数据收集与标注工具 - 针对盲人导航场景优化
支持自定义类别和地面视角数据收集
"""
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from typing import List, Tuple, Dict

# 盲人导航专用障碍物类别
OBSTACLE_CATEGORIES = {
    0: 'person',           # 行人
    1: 'bicycle',          # 自行车
    2: 'motorcycle',       # 摩托车/电动车
    3: 'car',              # 汽车
    4: 'pole',             # 电线杆/路灯杆
    5: 'tree',             # 树木
    6: 'bench',            # 长椅
    7: 'trash_can',        # 垃圾桶
    8: 'stairs_up',        # 上行台阶
    9: 'stairs_down',      # 下行台阶
    10: 'pothole',         # 路面坑洞
    11: 'speed_bump',      # 减速带
    12: 'wire',            # 低垂电线
    13: 'construction',    # 施工围栏
    14: 'door',            # 门/玻璃门
}


class ObstacleDataCollector:
    """
    障碍物数据收集器 - 支持手动标注和自动收集
    """
    
    def __init__(self, base_dir: str = 'training_data/obstacle_custom'):
        self.base_dir = base_dir
        self.images_dir = f'{base_dir}/images'
        self.labels_dir = f'{base_dir}/labels'
        self.meta_dir = f'{base_dir}/metadata'
        
        # 创建目录
        for d in [self.images_dir, self.labels_dir, self.meta_dir]:
            os.makedirs(d, exist_ok=True)
        
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.collected_count = 0
        
        # 保存类别定义
        self._save_categories()
        
        print(f"[障碍物数据收集器] 会话ID: {self.session_id}")
        print(f"  保存路径: {self.base_dir}")
        print(f"  类别数: {len(OBSTACLE_CATEGORIES)}")
    
    def _save_categories(self):
        """保存类别定义文件"""
        categories_path = f'{self.base_dir}/categories.json'
        with open(categories_path, 'w', encoding='utf-8') as f:
            json.dump(OBSTACLE_CATEGORIES, f, indent=2, ensure_ascii=False)
        print(f"  类别定义已保存: {categories_path}")
    
    def capture_from_camera(self, camera_id: int = 0, num_samples: int = 10):
        """
        从摄像头实时采集数据并手动标注
        
        Args:
            camera_id: 摄像头ID
            num_samples: 要采集的样本数
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"[错误] 无法打开摄像头 {camera_id}")
            return
        
        print("\n[摄像头采集模式]")
        print("操作说明:")
        print("  SPACE - 拍照并标注")
        print("  S - 保存当前帧（自动检测障碍物）")
        print("  Q - 退出")
        print("-" * 60)
        
        sample_count = 0
        
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 显示实时画面
            display = frame.copy()
            cv2.putText(display, f"Sample: {sample_count}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "SPACE: Capture | S: Auto-detect | Q: Quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Obstacle Data Collection', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                # 手动标注模式
                self._manual_annotate(frame)
                sample_count += 1
            elif key == ord('s'):
                # 自动检测并保存
                self._auto_detect_and_save(frame)
                sample_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[采集完成] 共保存 {sample_count} 个样本")
    
    def _manual_annotate(self, image: np.ndarray):
        """
        手动标注单张图片
        支持绘制边界框和选择类别
        """
        print("\n[手动标注模式]")
        print("类别列表:")
        for id, name in OBSTACLE_CATEGORIES.items():
            print(f"  {id}: {name}")
        
        annotations = []
        temp_image = image.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, temp_image
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp_image = image.copy()
                cv2.rectangle(temp_image, start_point, (x, y), (0, 255, 0), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                end_point = (x, y)
                
                # 计算边界框
                x1, y1 = start_point
                x2, y2 = end_point
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                
                if w > 20 and h > 20:  # 过滤太小的框
                    # 选择类别
                    print(f"\n绘制边界框: ({x}, {y}, {w}, {h})")
                    class_id = input("输入类别ID (0-14): ").strip()
                    try:
                        class_id = int(class_id)
                        if class_id in OBSTACLE_CATEGORIES:
                            annotations.append((class_id, x, y, w, h))
                            # 绘制确认
                            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            label = OBSTACLE_CATEGORIES[class_id]
                            cv2.putText(image, f"{class_id}:{label}", 
                                      (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            print(f"  已添加: {label}")
                        else:
                            print("  无效的类别ID")
                    except ValueError:
                        print("  请输入数字")
        
        drawing = False
        start_point = None
        
        cv2.namedWindow('Manual Annotation')
        cv2.setMouseCallback('Manual Annotation', mouse_callback)
        
        print("\n操作说明:")
        print("  鼠标左键拖拽 - 绘制边界框")
        print("  C - 完成当前图片")
        print("  R - 重置所有标注")
        print("  S - 跳过此图片")
        
        while True:
            cv2.imshow('Manual Annotation', temp_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                if annotations:
                    self._save_sample(image, annotations)
                    break
                else:
                    print("请先标注至少一个障碍物")
            elif key == ord('r'):
                annotations = []
                temp_image = image.copy()
                print("已重置标注")
            elif key == ord('s'):
                print("跳过此图片")
                break
        
        cv2.destroyWindow('Manual Annotation')
    
    def _auto_detect_and_save(self, image: np.ndarray):
        """
        使用当前模型自动检测障碍物并保存
        用户可以确认或修改检测结果
        """
        print("\n[自动检测模式]")
        
        # 这里可以集成现有的障碍物检测器
        # 暂时使用简单的边缘检测作为示例
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotations = []
        h, w = image.shape[:2]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # 过滤小区域
                x, y, bw, bh = cv2.boundingRect(cnt)
                # 默认类别为 obstacle (0)
                annotations.append((0, x, y, bw, bh))
        
        if annotations:
            print(f"自动检测到 {len(annotations)} 个潜在障碍物")
            # 显示检测结果
            display = image.copy()
            for class_id, x, y, bw, bh in annotations:
                cv2.rectangle(display, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            
            cv2.imshow('Auto Detection', display)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow('Auto Detection')
            
            if key == ord('y'):
                self._save_sample(image, annotations)
                print("已保存")
            else:
                print("已取消")
        else:
            print("未检测到明显障碍物")
    
    def _save_sample(self, image: np.ndarray, annotations: List[Tuple]):
        """
        保存样本图片和标注
        
        Args:
            image: 原始图片
            annotations: 标注列表 [(class_id, x, y, w, h), ...]
        """
        timestamp = datetime.now().strftime('%H%M%S_%f')
        filename = f'obstacle_{self.session_id}_{timestamp}'
        
        # 保存图片
        img_path = f'{self.images_dir}/{filename}.jpg'
        cv2.imwrite(img_path, image)
        
        # 保存YOLO格式标注（归一化）
        h, w = image.shape[:2]
        label_path = f'{self.labels_dir}/{filename}.txt'
        
        with open(label_path, 'w') as f:
            for class_id, x, y, bw, bh in annotations:
                x_center = (x + bw/2) / w
                y_center = (y + bh/2) / h
                width = bw / w
                height = bh / h
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # 保存元数据
        meta = {
            'timestamp': timestamp,
            'filename': filename,
            'image_size': {'width': w, 'height': h},
            'num_objects': len(annotations),
            'objects': [
                {
                    'class_id': int(a[0]),
                    'class_name': OBSTACLE_CATEGORIES.get(int(a[0]), 'unknown'),
                    'bbox': {'x': int(a[1]), 'y': int(a[2]), 'w': int(a[3]), 'h': int(a[4])}
                }
                for a in annotations
            ]
        }
        
        meta_path = f'{self.meta_dir}/{filename}.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        self.collected_count += 1
        print(f"[保存成功] {filename} ({len(annotations)} 个标注)")
    
    def import_existing_images(self, image_dir: str):
        """
        导入已有图片进行批量标注
        
        Args:
            image_dir: 图片目录路径
        """
        image_files = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
        
        print(f"\n[批量导入] 找到 {len(image_files)} 张图片")
        
        for img_file in image_files:
            print(f"\n处理: {img_file.name}")
            image = cv2.imread(str(img_file))
            if image is not None:
                self._manual_annotate(image)
            else:
                print(f"  无法读取图片")
        
        print(f"\n[导入完成] 共处理 {len(image_files)} 张图片")
    
    def import_coco_dataset(self, coco_images_dir: str, coco_labels_dir: str):
        """
        导入 COCO 数据集（已转换为 YOLO 格式）
        
        Args:
            coco_images_dir: COCO 图片目录
            coco_labels_dir: COCO YOLO 标注目录
        """
        import shutil
        
        coco_images = Path(coco_images_dir)
        coco_labels = Path(coco_labels_dir)
        
        if not coco_images.exists():
            print(f"[错误] COCO 图片目录不存在: {coco_images}")
            return
        
        if not coco_labels.exists():
            print(f"[错误] COCO 标注目录不存在: {coco_labels}")
            return
        
        # 获取所有标注文件
        label_files = list(coco_labels.glob('*.txt'))
        print(f"\n[导入 COCO 数据] 找到 {len(label_files)} 个标注文件")
        
        imported_count = 0
        skipped_count = 0
        
        for label_file in label_files:
            # 对应的图片文件
            img_name = label_file.stem + '.jpg'
            img_path = coco_images / img_name
            
            if not img_path.exists():
                # 尝试 png 格式
                img_path = coco_images / (label_file.stem + '.png')
                if not img_path.exists():
                    skipped_count += 1
                    continue
            
            # 复制图片
            dst_img = Path(self.images_dir) / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            
            # 复制标注
            dst_label = Path(self.labels_dir) / label_file.name
            if not dst_label.exists():
                shutil.copy2(label_file, dst_label)
            
            imported_count += 1
            
            if imported_count % 1000 == 0:
                print(f"  已导入 {imported_count} 张...")
        
        print(f"\n[导入完成] 成功: {imported_count}, 跳过: {skipped_count}")
        self.collected_count += imported_count
    
    def get_stats(self) -> Dict:
        """获取收集统计信息"""
        return {
            'session_id': self.session_id,
            'total_samples': self.collected_count,
            'images_dir': self.images_dir,
            'labels_dir': self.labels_dir
        }
    
    def preview_dataset(self):
        """预览已收集的数据集"""
        label_files = list(Path(self.labels_dir).glob('*.txt'))
        
        print("\n[数据集预览]")
        print(f"总样本数: {len(label_files)}")
        
        class_counts = {}
        for label_file in label_files:
            with open(label_file) as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        print("\n类别分布:")
        for class_id, count in sorted(class_counts.items()):
            name = OBSTACLE_CATEGORIES.get(class_id, 'unknown')
            print(f"  {class_id}: {name} - {count} 个")


def quick_start_guide():
    """
    快速开始指南
    
    步骤1: 收集数据
        python obstacle_data_collector.py
        选择模式1，使用摄像头采集障碍物图片并手动标注
    
    步骤2: 验证数据
        python obstacle_data_collector.py
        选择模式3，预览已收集的数据集
    
    步骤3: 训练模型
        python train_obstacle_model_custom.py --epochs 50
    
    步骤4: 部署模型
        训练完成后，模型将保存到 assets/models/yolov8_obstacle_custom.pth
        修改 config.py 中的模型路径即可使用新模型
    
    注意事项:
    - 建议每类收集至少50-100张图片
    - 图片应从地面视角拍摄（模拟导盲杖高度）
    - 标注时确保边界框紧密包围障碍物
    """
    pass


def main():
    """主函数 - 交互式数据收集"""
    print("=" * 70)
    print("障碍物数据收集工具 - 盲人导航专用")
    print("=" * 70)
    print("\n快速指南:")
    print("  1. 摄像头采集 - 实时拍摄并手动标注")
    print("  2. 导入已有图片 - 批量标注已有图片")
    print("  3. 导入 COCO 数据集 - 导入已转换的 COCO 数据")
    print("  4. 预览数据 - 查看已收集的数据集")
    print("\n提示: 每类建议收集50-100张图片，从地面视角拍摄")
    print("-" * 70)
    
    collector = ObstacleDataCollector()
    
    print("\n选择模式:")
    print("  1. 摄像头实时采集")
    print("  2. 导入已有图片")
    print("  3. 导入 COCO 数据集")
    print("  4. 预览数据集")
    print("  5. 退出")
    
    choice = input("\n输入选项 (1-5): ").strip()
    
    if choice == '1':
        num = int(input("要采集的样本数 (默认10): ") or "10")
        collector.capture_from_camera(num_samples=num)
    elif choice == '2':
        img_dir = input("图片目录路径: ").strip()
        if os.path.exists(img_dir):
            collector.import_existing_images(img_dir)
        else:
            print("目录不存在")
    elif choice == '3':
        print("\n[导入 COCO 数据集]")
        print("请先运行: python download_coco_for_obstacle.py")
        print("下载并转换 COCO 数据后，再使用此选项导入")
        print()
        coco_img_dir = input("COCO 图片目录 (如: training_data/coco_for_obstacle/images/train): ").strip()
        coco_label_dir = input("COCO 标注目录 (如: training_data/coco_for_obstacle/labels/train): ").strip()
        if os.path.exists(coco_img_dir) and os.path.exists(coco_label_dir):
            collector.import_coco_dataset(coco_img_dir, coco_label_dir)
        else:
            print("目录不存在，请先下载 COCO 数据")
    elif choice == '4':
        collector.preview_dataset()
    else:
        print("退出")
    
    print("\n统计:", collector.get_stats())


if __name__ == '__main__':
    main()
