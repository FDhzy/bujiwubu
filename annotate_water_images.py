"""
水坑图片标注工具
使用OpenCV进行交互式标注，生成掩码文件
"""
import os
import cv2
import numpy as np
from pathlib import Path

class WaterAnnotator:
    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有未标注的图片
        self.images = self._get_unannotated_images()
        self.current_idx = 0
        
        # 标注状态
        self.drawing = False
        self.current_mask = None
        self.display_image = None
        self.brush_size = 20
        
    def _get_unannotated_images(self):
        """获取所有未标注的图片"""
        all_images = list(self.image_dir.glob("*.jpg")) + \
                     list(self.image_dir.glob("*.png"))
        
        unannotated = []
        for img_path in all_images:
            mask_path = self.mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                unannotated.append(img_path)
        
        return sorted(unannotated)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv2.circle(self.current_mask, (x, y), self.brush_size, 255, -1)
            cv2.circle(self.display_image, (x, y), self.brush_size, (0, 255, 0), -1)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            cv2.circle(self.current_mask, (x, y), self.brush_size, 255, -1)
            cv2.circle(self.display_image, (x, y), self.brush_size, (0, 255, 0), -1)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
    
    def _create_mask_from_image(self, image_path):
        """创建掩码标注"""
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"错误：无法读取图片 {image_path}")
            return None
        
        # 调整大小以便显示
        max_size = 800
        h, w = image.shape[:2]
        scale = min(max_size / w, max_size / h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # 初始化掩码和显示图像
        self.current_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        self.display_image = image.copy()
        
        # 创建窗口
        window_name = f"标注水坑区域 - {image_path.name}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        print(f"\n正在标注: {image_path.name}")
        print("操作说明:")
        print("  左键拖动: 标注水坑区域")
        print("  +/-: 调整画笔大小")
        print("  r: 重置标注")
        print("  s: 保存并下一张")
        print("  q: 跳过此图")
        print("  ESC: 退出标注")
        
        while True:
            # 显示叠加效果
            overlay = self.display_image.copy()
            mask_colored = np.zeros_like(overlay)
            mask_colored[self.current_mask > 0] = [0, 0, 255]  # 红色标记
            cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0, overlay)
            
            # 显示画笔大小
            cv2.putText(overlay, f"Brush: {self.brush_size}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow(window_name, overlay)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC - 退出
                cv2.destroyWindow(window_name)
                return "exit"
                
            elif key == ord('s'):  # 保存
                # 恢复原尺寸的掩码
                original_mask = cv2.resize(self.current_mask, (w, h))
                mask_path = self.mask_dir / f"{image_path.stem}.png"
                cv2.imwrite(str(mask_path), original_mask)
                print(f"  ✓ 已保存: {mask_path.name}")
                cv2.destroyWindow(window_name)
                return "saved"
                
            elif key == ord('q'):  # 跳过
                print(f"  → 跳过: {image_path.name}")
                cv2.destroyWindow(window_name)
                return "skipped"
                
            elif key == ord('r'):  # 重置
                self.current_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                self.display_image = image.copy()
                print("  ↺ 重置标注")
                
            elif key == ord('+'):  # 增大画笔
                self.brush_size = min(50, self.brush_size + 5)
                
            elif key == ord('-'):  # 减小画笔
                self.brush_size = max(5, self.brush_size - 5)
    
    def run(self):
        """运行标注工具"""
        print("=" * 60)
        print("水坑图片标注工具")
        print("=" * 60)
        
        if not self.images:
            print("\n✓ 所有图片都已标注完成！")
            return
        
        print(f"\n找到 {len(self.images)} 张未标注的图片")
        print(f"已标注: {len(list(self.mask_dir.glob('*.png')))} 张")
        
        annotated_count = 0
        skipped_count = 0
        
        for i, img_path in enumerate(self.images):
            print(f"\n进度: [{i+1}/{len(self.images)}] 剩余: {len(self.images)-i-1}")
            
            result = self._create_mask_from_image(img_path)
            
            if result == "exit":
                break
            elif result == "saved":
                annotated_count += 1
            elif result == "skipped":
                skipped_count += 1
        
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("标注完成！")
        print(f"本次标注: {annotated_count} 张")
        print(f"跳过: {skipped_count} 张")
        print(f"总计已标注: {len(list(self.mask_dir.glob('*.png')))} 张")
        print("=" * 60)

def quick_auto_mask(image_dir, mask_dir):
    """
    快速自动创建初始掩码（基于颜色检测，需要后续手动修正）
    """
    print("=" * 60)
    print("快速自动标注模式")
    print("=" * 60)
    print("基于颜色检测自动识别水坑区域（需要后续手动修正）")
    
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取未标注的图片
    all_images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    unannotated = []
    for img_path in all_images:
        mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            unannotated.append(img_path)
    
    print(f"\n找到 {len(unannotated)} 张未标注的图片")
    
    created = 0
    for img_path in unannotated:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # 转换为HSV进行颜色分析
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 水坑通常呈现深色、低饱和度的特征
        # 这里使用简单的阈值作为初始掩码
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值检测暗区域（可能是水坑）
        mask = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        
        # 形态学操作清理噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 保存初始掩码
        mask_path = mask_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(mask_path), mask)
        created += 1
        
        if created % 10 == 0:
            print(f"  已创建 {created}/{len(unannotated)} 个初始掩码...")
    
    print(f"\n✓ 自动创建了 {created} 个初始掩码")
    print("注意：这些掩码需要手动检查并修正")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="水坑图片标注工具")
    parser.add_argument("--auto", action="store_true", 
                       help="使用自动模式创建初始掩码（需要后续手动修正）")
    
    args = parser.parse_args()
    
    image_dir = "training_data/water_real/images"
    mask_dir = "training_data/water_real/masks"
    
    if args.auto:
        quick_auto_mask(image_dir, mask_dir)
    else:
        annotator = WaterAnnotator(image_dir, mask_dir)
        annotator.run()
