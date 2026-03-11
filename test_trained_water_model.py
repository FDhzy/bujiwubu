"""
测试训练好的水坑检测模型
"""
import sys
import numpy as np
import cv2

sys.path.insert(0, '.')

from models.pytorch_water_detector import PyTorchWaterDetector


def test_trained_model():
    """测试训练好的模型"""
    print("=" * 60)
    print("测试训练好的水坑检测模型")
    print("=" * 60)
    
    # 加载训练好的模型
    detector = PyTorchWaterDetector(
        model_path="training_checkpoints/water_model_best.pth",
        threshold=0.5
    )
    
    # 测试1: 合成水坑
    print("\n[测试1] 合成水坑场景")
    test_image1 = np.random.randint(50, 100, (480, 640, 3), dtype=np.uint8)
    cv2.ellipse(test_image1, (320, 240), (100, 60), 0, 0, 360, (80, 85, 90), -1)
    
    result1 = detector.detect(test_image1)
    if result1:
        print(f"  覆盖率: {result1.coverage_ratio:.2%}")
        print(f"  区域数: {result1.num_regions}")
        depth1 = detector.estimate_depth(test_image1, result1.mask)
        print(f"  估算水深: {depth1:.2f}")
    
    # 测试2: 无水面场景
    print("\n[测试2] 无水面场景")
    test_image2 = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    
    result2 = detector.detect(test_image2)
    if result2:
        print(f"  覆盖率: {result2.coverage_ratio:.2%}")
        print(f"  区域数: {result2.num_regions}")
    
    # 测试3: 多个水坑
    print("\n[测试3] 多个水坑场景")
    test_image3 = np.random.randint(60, 110, (480, 640, 3), dtype=np.uint8)
    cv2.ellipse(test_image3, (200, 200), (60, 40), 30, 0, 360, (75, 80, 85), -1)
    cv2.ellipse(test_image3, (450, 300), (80, 50), -20, 0, 360, (70, 75, 80), -1)
    cv2.ellipse(test_image3, (350, 150), (50, 35), 45, 0, 360, (72, 77, 82), -1)
    
    result3 = detector.detect(test_image3)
    if result3:
        print(f"  覆盖率: {result3.coverage_ratio:.2%}")
        print(f"  区域数: {result3.num_regions}")
        depth3 = detector.estimate_depth(test_image3, result3.mask)
        print(f"  估算水深: {depth3:.2f}")
        
        # 保存可视化结果
        vis = test_image3.copy()
        mask_colored = np.zeros_like(vis)
        mask_colored[result3.mask > 0] = [0, 0, 255]
        vis = cv2.addWeighted(vis, 0.6, mask_colored, 0.4, 0)
        cv2.imwrite("water_detection_multi.jpg", vis)
        print(f"  可视化结果已保存: water_detection_multi.jpg")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_trained_model()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
