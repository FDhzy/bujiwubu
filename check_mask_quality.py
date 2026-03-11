"""
抽样检查掩码质量
随机显示几张图片及其自动生成的掩码
"""
import cv2
import numpy as np
from pathlib import Path
import random

def check_mask_quality(sample_count=5):
    """随机抽样检查掩码质量"""
    print("=" * 60)
    print("掩码质量抽样检查")
    print("=" * 60)
    
    image_dir = Path("training_data/water_real/images")
    mask_dir = Path("training_data/water_real/masks")
    
    # 获取所有图片
    all_images = list(image_dir.glob("*.jpg"))
    
    # 随机选择样本
    samples = random.sample(all_images, min(sample_count, len(all_images)))
    
    print(f"\n随机选择 {len(samples)} 张图片进行检查")
    print("按键说明：")
    print("  n - 下一张")
    print("  q - 退出检查")
    print("-" * 60)
    
    for i, img_path in enumerate(samples):
        # 读取图片
        img_bytes = np.fromfile(str(img_path), dtype=np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        # 读取掩码
        mask_path = mask_dir / f"{img_path.stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            print(f"  无法读取: {img_path.name}")
            continue
        
        # 调整大小以便显示
        max_size = 600
        h, w = image.shape[:2]
        scale = min(max_size / w, max_size / h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h))
        
        # 创建叠加显示
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = [0, 0, 255]  # 红色标记掩码区域
        overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
        
        # 显示信息
        info_text = f"[{i+1}/{len(samples)}] {img_path.name}"
        cv2.putText(overlay, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, "Press 'n' for next, 'q' to quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 并排显示原图和叠加图
        comparison = np.hstack([image, overlay])
        
        print(f"  显示: {img_path.name}")
        
        cv2.imshow("Mask Quality Check (Original | Overlay)", comparison)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):  # 下一张
                break
            elif key == ord('q'):  # 退出
                cv2.destroyAllWindows()
                print("\n检查已退出")
                return
    
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("抽样检查完成！")
    print("=" * 60)
    print("\n基于抽样结果，你可以选择：")
    print("1. 直接开始训练（如果掩码质量看起来还可以）")
    print("2. 使用 annotate_water_images.py 手动修正")

if __name__ == "__main__":
    check_mask_quality(sample_count=5)
