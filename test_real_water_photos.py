"""
使用真实照片测试水坑检测模型
"""
import sys
import cv2
from pathlib import Path

sys.path.insert(0, '.')

from models.pytorch_water_detector import PyTorchWaterDetector


def test_real_photos():
    """测试真实水坑照片"""
    print("=" * 60)
    print("使用真实照片测试水坑检测模型")
    print("=" * 60)
    
    # 加载训练好的模型
    detector = PyTorchWaterDetector(
        model_path="training_checkpoints/water_model_best.pth",
        threshold=0.15  # 与主程序保持一致
    )
    
    # 获取真实照片
    image_dir = Path("training_data/water_real/images")
    real_photos = sorted(image_dir.glob("real_puddle_*.jpg"))[:5]  # 测试前5张
    
    print(f"\n测试 {len(real_photos)} 张真实照片")
    print("-" * 60)
    
    for i, photo_path in enumerate(real_photos, 1):
        # 加载图片
        image = cv2.imread(str(photo_path))
        if image is None:
            continue
        
        # 检测水坑
        result = detector.detect(image)
        
        if result:
            print(f"\n[{i}] {photo_path.name}")
            print(f"  覆盖率: {result.coverage_ratio:.2%}")
            print(f"  区域数: {result.num_regions}")
            
            # 估算水深
            depth = detector.estimate_depth(image, result.mask)
            print(f"  估算水深: {depth:.2f}")
            
            # 保存可视化结果
            if result.coverage_ratio > 0.01:  # 只保存检测到有水的
                vis = image.copy()
                mask_colored = vis.copy()
                mask_colored[result.mask > 0] = [0, 0, 255]
                vis = cv2.addWeighted(vis, 0.6, mask_colored, 0.4, 0)
                
                output_path = f"water_test_{photo_path.stem}.jpg"
                cv2.imwrite(output_path, vis)
                print(f"  ✓ 保存结果: {output_path}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_real_photos()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
