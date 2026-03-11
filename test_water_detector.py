"""
测试改进的水坑检测器
"""
import sys
import numpy as np
import cv2

sys.path.insert(0, '.')

from models.pytorch_water_detector import PyTorchWaterDetector


def test_water_detector():
    """测试水坑检测器"""
    print("=" * 60)
    print("测试改进的水坑检测器")
    print("=" * 60)
    
    # 创建检测器（不加载权重，使用随机初始化进行测试）
    detector = PyTorchWaterDetector(model_path="training_checkpoints/water_model_best.pth", threshold=0.5)
    
    # 创建测试图像 - 模拟水坑
    print("\n生成测试图像...")
    test_image = np.random.randint(80, 150, (480, 640, 3), dtype=np.uint8)
    
    # 添加一个椭圆形的"水坑"
    center = (320, 240)
    axes = (100, 60)
    cv2.ellipse(test_image, center, axes, 0, 0, 360, (100, 100, 120), -1)
    
    print("运行检测...")
    result = detector.detect(test_image)
    
    if result:
        print(f"\n检测结果:")
        print(f"  覆盖率: {result.coverage_ratio:.2%}")
        print(f"  区域数: {result.num_regions}")
        print(f"  平均置信度: {np.mean(result.confidence):.4f}")
        
        # 估算水深
        depth = detector.estimate_depth(test_image, result.mask)
        print(f"  估算水深: {depth:.2f}")
        
        # 可视化结果
        vis_image = test_image.copy()
        mask_colored = np.zeros_like(vis_image)
        mask_colored[result.mask > 0] = [0, 0, 255]  # 红色标记水坑
        vis_image = cv2.addWeighted(vis_image, 0.7, mask_colored, 0.3, 0)
        
        # 保存结果
        cv2.imwrite("water_detection_result.jpg", vis_image)
        print(f"\n可视化结果已保存: water_detection_result.jpg")
        
        print("\n" + "=" * 60)
        print("测试完成!")
        print("=" * 60)
        
        return True
    else:
        print("检测失败!")
        return False


if __name__ == "__main__":
    try:
        success = test_water_detector()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
