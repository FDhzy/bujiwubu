"""
简化版水坑标注工具
操作更简单，只需点击水坑区域自动生成mask
"""
import cv2
import numpy as np
from pathlib import Path
import sys


class SimpleWaterAnnotator:
    """简化版水坑标注器"""
    
    def __init__(self):
        self.image_dir = Path("training_data/water_real/images")
        self.mask_dir = Path("training_data/water_real/masks")
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_image = None
        self.current_mask = None
        self.display_image = None
        self.clicked_points = []
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键点击添加水坑中心点
            self.clicked_points.append((x, y))
            # 绘制标记
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(self.display_image, (x, y), 20, (0, 255, 0), 2)
            print(f"  添加水坑中心点: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键点击清除所有点
            self.clicked_points = []
            self.display_image = self.current_image.copy()
            print("  清除所有标记")
    
    def generate_mask_from_points(self, image_shape):
        """根据点击的点生成水坑mask"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        for point in self.clicked_points:
            # 在点击位置生成椭圆水坑
            center = point
            # 随机大小（可根据需要调整）
            axes = (np.random.randint(30, 80), np.random.randint(20, 60))
            angle = np.random.randint(0, 180)
            cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        
        return mask
    
    def annotate_image(self, image_path):
        """标注单张图片"""
        # 加载图片
        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            print(f"无法加载图片: {image_path}")
            return False
        
        self.display_image = self.current_image.copy()
        self.clicked_points = []
        
        # 创建窗口
        window_name = f"标注: {image_path.name}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print(f"\n正在标注: {image_path.name}")
        print("-" * 60)
        print("操作说明:")
        print("  左键点击: 标记水坑中心位置")
        print("  右键点击: 清除所有标记")
        print("  按 's'   : 保存标注")
        print("  按 'q'   : 跳过此图")
        print("  按 'r'   : 重新标注")
        print("-" * 60)
        
        while True:
            # 显示提示文字
            display = self.display_image.copy()
            cv2.putText(display, f"Points: {len(self.clicked_points)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Left: add | Right: clear | S: save | Q: skip", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord('S'):  # 保存（支持大小写）
                if len(self.clicked_points) > 0:
                    # 生成mask
                    mask = self.generate_mask_from_points(self.current_image.shape)
                    
                    # 保存mask
                    mask_path = self.mask_dir / image_path.name.replace('.jpg', '.png')
                    success = cv2.imwrite(str(mask_path), mask)
                    
                    if success:
                        print(f"  ✓ 已保存标注: {mask_path.name}")
                        print(f"    标记了 {len(self.clicked_points)} 个水坑区域")
                    else:
                        print(f"  ✗ 保存失败: {mask_path.name}")
                else:
                    print("  ! 没有标记任何水坑，跳过保存")
                
                # 短暂延迟确保保存完成
                cv2.waitKey(100)
                break
                
            elif key == ord('q'):  # 跳过
                print("  - 跳过此图")
                break
                
            elif key == ord('r'):  # 重新标注
                self.clicked_points = []
                self.display_image = self.current_image.copy()
                print("  重新标注")
        
        cv2.destroyWindow(window_name)
        return True
    
    def run(self, target_files=None, max_images=100, reannotate=False):
        """运行标注工具"""
        print("=" * 60)
        print("简化版水坑标注工具")
        print("=" * 60)
        
        # 获取要标注的图片
        if target_files:
            # 只标注指定文件
            image_files = [self.image_dir / f for f in target_files if (self.image_dir / f).exists()]
        else:
            # 获取所有jpg图片（包括所有命名格式）
            all_images = sorted(self.image_dir.glob("*.jpg"))
            
            if reannotate:
                # 重新标注模式：处理所有图片（包括已有掩码的）
                image_files = all_images
                print("重新标注模式：将处理所有图片（包括已有掩码）")
            else:
                # 过滤出没有mask的图片
                image_files = []
                for img_path in all_images:
                    mask_path = self.mask_dir / img_path.name.replace('.jpg', '.png')
                    if not mask_path.exists():
                        image_files.append(img_path)
            
            # 限制数量
            if len(image_files) > max_images:
                import random
                random.shuffle(image_files)
                image_files = image_files[:max_images]
                print(f"随机选择 {max_images} 张图片进行标注")
        
        if not image_files:
            print("没有找到需要标注的图片!")
            print(f"请确保图片在: {self.image_dir}")
            return
        
        print(f"\n找到 {len(image_files)} 张真实水坑照片需要标注")
        print("-" * 60)
        
        # 开始标注
        completed = 0
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] ", end="")
            self.annotate_image(img_path)
            completed += 1
        
        print("\n" + "=" * 60)
        print(f"标注完成! 共处理 {completed} 张图片")
        print(f"标注文件保存在: {self.mask_dir}")
        print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="简化版水坑标注工具")
    parser.add_argument("--max-images", type=int, default=50,
                       help="最多处理多少张图片（默认50）")
    parser.add_argument("--reannotate", action="store_true",
                       help="重新标注模式（包括已有掩码的图片）")
    
    args = parser.parse_args()
    
    annotator = SimpleWaterAnnotator()
    annotator.run(max_images=args.max_images, reannotate=args.reannotate)
