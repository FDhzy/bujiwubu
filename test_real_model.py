"""
测试真实照片训练的鞋子识别模型
"""
import sys
import numpy as np
import cv2

# 添加项目路径
sys.path.insert(0, '.')

try:
    from models.pytorch_shoe_model import PyTorchShoeModel
    
    print("=" * 50)
    print("测试真实照片鞋子识别模型")
    print("=" * 50)
    
    # 创建模型
    model = PyTorchShoeModel(use_database=False)
    
    # 加载训练好的模型
    success = model.load_model('training_checkpoints/shoe_model_best.pth')
    print(f"\n模型加载: {'成功' if success else '失败'}")
    
    if success:
        # 测试预测
        print("\n测试预测...")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        shoe_type, confidence = model.predict(test_image)
        print(f"预测结果: {shoe_type.value}")
        print(f"置信度: {confidence:.2%}")
        
        # 测试不同颜色的图片
        print("\n测试不同特征的图片...")
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            test_img = np.full((224, 224, 3), color, dtype=np.uint8)
            st, conf = model.predict(test_img)
            print(f"  测试 {i+1}: {st.value} ({conf:.2%})")
        
        print("\n" + "=" * 50)
        print("模型测试完成!")
        print("=" * 50)
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
