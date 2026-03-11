"""
水坑/积水数据收集工具
用于收集和标注水坑图片，训练改进的检测模型
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')


class WaterDataCollector:
    """水坑数据收集器"""
    
    def __init__(self):
        self.dataset_dir = Path("training_data/water_real")
        self.image_dir = self.dataset_dir / "images"
        self.mask_dir = self.dataset_dir / "masks"
        
        # 创建目录
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        
        self.drawing = False
        self.mask = None
        self.current_image = None
        self.image_name = None
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数 - 绘制水坑区域"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # 绘制水坑区域（白色）
            cv2.circle(self.mask, (x, y), 10, 255, -1)
            cv2.circle(self.display_image, (x, y), 10, (0, 255, 0), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
    
    def annotate_image(self, image_path):
        """标注单张图片"""
        # 加载图片
        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            print(f"无法加载图片: {image_path}")
            return False
        
        h, w = self.current_image.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self.display_image = self.current_image.copy()
        
        # 创建窗口
        cv2.namedWindow("Water Annotation")
        cv2.setMouseCallback("Water Annotation", self.mouse_callback)
        
        print("\n标注说明:")
        print("- 按住鼠标左键绘制水坑区域")
        print("- 按 's' 保存标注")
        print("- 按 'c' 清除标注")
        print("- 按 'q' 跳过此图")
        
        while True:
            cv2.imshow("Water Annotation", self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # 保存
                # 保存原始图片
                save_name = f"water_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000):03d}.jpg"
                cv2.imwrite(str(self.image_dir / save_name), self.current_image)
                cv2.imwrite(str(self.mask_dir / save_name.replace('.jpg', '.png')), self.mask)
                print(f"  已保存: {save_name}")
                break
            
            elif key == ord('c'):  # 清除
                self.mask = np.zeros((h, w), dtype=np.uint8)
                self.display_image = self.current_image.copy()
                print("  已清除标注")
            
            elif key == ord('q'):  # 跳过
                print("  跳过此图")
                break
        
        cv2.destroyWindow("Water Annotation")
        return True
    
    def collect_from_camera(self):
        """从摄像头收集数据"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        print("\n摄像头数据收集模式")
        print("-" * 60)
        print("按 'c' 捕获当前帧并标注")
        print("按 'q' 退出")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 显示实时画面
            display = frame.copy()
            cv2.putText(display, "Press 'c' to capture, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Water Data Collection", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # 捕获
                # 保存临时图片
                temp_path = self.image_dir / "temp_capture.jpg"
                cv2.imwrite(str(temp_path), frame)
                self.annotate_image(temp_path)
                if temp_path.exists():
                    temp_path.unlink()  # 删除临时文件
            
            elif key == ord('q'):  # 退出
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def collect_from_folder(self, source_folder):
        """从文件夹批量处理图片"""
        source_path = Path(source_folder)
        if not source_path.exists():
            print(f"文件夹不存在: {source_folder}")
            return
        
        # 获取所有图片
        image_files = list(source_path.glob("*.jpg")) + \
                      list(source_path.glob("*.png"))
        
        print(f"\n找到 {len(image_files)} 张图片")
        print("-" * 60)
        
        for i, img_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] 处理: {img_path.name}")
            self.annotate_image(img_path)
        
        print("\n批量处理完成!")
    
    def generate_synthetic_data(self, num_samples=100):
        """生成合成水坑数据"""
        print(f"\n生成 {num_samples} 张合成水坑图片...")
        
        for i in range(num_samples):
            # 创建背景（模拟地面）
            bg = np.random.randint(80, 150, (256, 256, 3), dtype=np.uint8)
            
            # 添加纹理
            noise = np.random.randint(-20, 20, (256, 256, 3), dtype=np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # 创建水坑mask
            mask = np.zeros((256, 256), dtype=np.uint8)
            
            # 随机生成1-3个水坑
            num_puddles = np.random.randint(1, 4)
            for _ in range(num_puddles):
                # 随机椭圆参数
                center = (np.random.randint(50, 206), np.random.randint(50, 206))
                axes = (np.random.randint(20, 60), np.random.randint(15, 40))
                angle = np.random.randint(0, 180)
                
                cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
                
                # 在水坑上添加反光效果
                cv2.ellipse(bg, center, axes, angle, 0, 360, 
                           (bg[center[1], center[0]] + 30).clip(0, 255).tolist(), -1)
            
            # 保存
            save_name = f"synthetic_water_{i:04d}.jpg"
            cv2.imwrite(str(self.image_dir / save_name), bg)
            cv2.imwrite(str(self.mask_dir / save_name.replace('.jpg', '.png')), mask)
            
            if (i + 1) % 20 == 0:
                print(f"  已生成: {i+1}/{num_samples}")
        
        print(f"✓ 合成数据生成完成: {num_samples} 张")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Water puddle data collector")
    parser.add_argument("--mode", choices=["camera", "folder", "synthetic"], 
                       default="synthetic", help="Collection mode")
    parser.add_argument("--source", type=str, help="Source folder for folder mode")
    parser.add_argument("--num-samples", type=int, default=100, 
                       help="Number of synthetic samples")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("水坑/积水数据收集工具")
    print("=" * 60)
    
    collector = WaterDataCollector()
    
    if args.mode == "camera":
        collector.collect_from_camera()
    elif args.mode == "folder":
        if not args.source:
            print("错误: 文件夹模式需要 --source 参数")
            return
        collector.collect_from_folder(args.source)
    elif args.mode == "synthetic":
        collector.generate_synthetic_data(args.num_samples)
    
    # 显示统计
    image_count = len(list(collector.image_dir.glob("*.jpg")))
    print(f"\n数据集统计:")
    print(f"  图片总数: {image_count}")
    print(f"  保存位置: {collector.dataset_dir}")


if __name__ == "__main__":
    main()
