"""
批量掩码修正工具
快速遍历图片，对需要修正的掩码进行重新标注
"""
import os
import cv2
import numpy as np
from pathlib import Path
import random

class BatchMaskAnnotator:
    def __init__(self, image_dir, mask_dir, max_images=100):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        
        # 优先处理没有掩码的图片，再补充随机其他图片
        all_images = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        missing = [p for p in all_images if not (self.mask_dir / f"{p.stem}.png").exists()]
        has_mask = [p for p in all_images if (self.mask_dir / f"{p.stem}.png").exists()]
        random.shuffle(has_mask)
        # 缺掩码的全部放前面，再补充部分已有掩码的图
        combined = missing + has_mask
        self.images = combined[:min(max_images, len(combined))]
        print(f"  → 缺少掩码: {len(missing)} 张（优先处理），已有掩码: {len(has_mask)} 张")
        
        self.current_idx = 0
        self.drawing = False
        self.brush_size = 25
        self.current_mask = None
        self.display_image = None
        self.original_image = None
        
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
            
        elif event == cv2.EVENT_RBUTTONDOWN:  # 右键擦除
            self.drawing = True
            cv2.circle(self.current_mask, (x, y), self.brush_size, 0, -1)
            # 恢复原图区域
            x1, y1 = max(0, x - self.brush_size), max(0, y - self.brush_size)
            x2, y2 = min(self.original_image.shape[1], x + self.brush_size), min(self.original_image.shape[0], y + self.brush_size)
            self.display_image[y1:y2, x1:x2] = self.original_image[y1:y2, x1:x2].copy()
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing and (flags & cv2.EVENT_FLAG_RBUTTON):
            cv2.circle(self.current_mask, (x, y), self.brush_size, 0, -1)
            x1, y1 = max(0, x - self.brush_size), max(0, y - self.brush_size)
            x2, y2 = min(self.original_image.shape[1], x + self.brush_size), min(self.original_image.shape[0], y + self.brush_size)
            self.display_image[y1:y2, x1:x2] = self.original_image[y1:y2, x1:x2].copy()
            
        elif event == cv2.EVENT_RBUTTONUP:
            self.drawing = False

    def annotate_image(self, img_path):
        """标注单张图片"""
        # 读取图片
        img_bytes = np.fromfile(str(img_path), dtype=np.uint8)
        original = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        if original is None:
            print(f"  ✗ 无法读取: {img_path.name}")
            return "skip"
        
        # 调整大小以便标注
        max_size = 800
        h, w = original.shape[:2]
        scale = min(max_size / w, max_size / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        self.original_image = cv2.resize(original, (new_w, new_h))
        
        # 尝试读取现有掩码
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            existing_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if existing_mask is not None:
                self.current_mask = cv2.resize(existing_mask, (new_w, new_h))
            else:
                self.current_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        else:
            self.current_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        
        # 初始化显示图像
        self.display_image = self.original_image.copy()
        
        # 如果有现有掩码，显示出来
        mask_overlay = np.zeros_like(self.display_image)
        mask_overlay[self.current_mask > 0] = [0, 0, 255]
        self.display_image = cv2.addWeighted(self.display_image, 0.7, mask_overlay, 0.3, 0)
        
        # 创建窗口（避免中文窗口名导致问题）
        window_name = f"Annotator [{self.current_idx + 1}/{len(self.images)}]"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        print(f"\n正在标注 [{self.current_idx + 1}/{len(self.images)}]: {img_path.name}")
        print("-" * 50)
        print("操作说明:")
        print("  左键拖动: 添加水坑区域（绿色）")
        print("  右键拖动: 擦除区域")
        print("  +/-: 调整画笔大小")
        print("  r: 重置为空白")
        print("  a: 接受并保存")
        print("  s: 跳过此图")
        print("  q: 退出标注")
        print("-" * 50)
        
        while True:
            # 显示当前状态
            display = self.display_image.copy()
            
            # 添加信息文字
            cv2.putText(display, f"Brush: {self.brush_size}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Progress: {self.current_idx + 1}/{len(self.images)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, "LMB:Draw  RMB:Erase  +/-:Size  A:Save  S:Skip  Q:Quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # 退出
                cv2.destroyWindow(window_name)
                return "quit"
                
            elif key == ord('a'):  # 接受并保存
                # 恢复原尺寸
                original_mask = cv2.resize(self.current_mask, (w, h))
                cv2.imwrite(str(mask_path), original_mask)
                print(f"  ✓ 已保存: {mask_path.name}")
                cv2.destroyWindow(window_name)
                return "saved"
                
            elif key == ord('s'):  # 跳过
                print(f"  → 跳过: {img_path.name}")
                cv2.destroyWindow(window_name)
                return "skipped"
                
            elif key == ord('r'):  # 重置
                self.current_mask = np.zeros((new_h, new_w), dtype=np.uint8)
                self.display_image = self.original_image.copy()
                print("  ↺ 已重置")
                
            elif key == ord('+'):  # 增大画笔
                self.brush_size = min(50, self.brush_size + 5)
                
            elif key == ord('-'):  # 减小画笔
                self.brush_size = max(5, self.brush_size - 5)
    
    def run(self):
        """运行批量标注"""
        print("=" * 60)
        print("批量掩码修正工具")
        print("=" * 60)
        print(f"\n将处理 {len(self.images)} 张图片")
        print("建议：优先修正质量较差或明显错误的掩码")
        print("=" * 60)
        
        saved_count = 0
        skipped_count = 0
        
        while self.current_idx < len(self.images):
            img_path = self.images[self.current_idx]
            result = self.annotate_image(img_path)
            
            if result == "quit":
                break
            elif result == "saved":
                saved_count += 1
            elif result == "skipped":
                skipped_count += 1
            
            self.current_idx += 1
        
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("标注完成！")
        print(f"已保存: {saved_count} 张")
        print(f"已跳过: {skipped_count} 张")
        print(f"剩余未处理: {len(self.images) - self.current_idx} 张")
        print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量掩码修正工具")
    parser.add_argument("--max-images", type=int, default=50,
                       help="最多处理多少张图片（默认50）")
    
    args = parser.parse_args()
    
    annotator = BatchMaskAnnotator(
        "training_data/water_real/images",
        "training_data/water_real/masks",
        max_images=args.max_images
    )
    annotator.run()
