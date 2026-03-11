"""
检查 ML 模型运行状态
"""
import os
import sys

# 检查模型文件是否存在
model_dir = 'assets/models'
deeplab_path = os.path.join(model_dir, 'deeplabv3_water.tflite')
yolo_path = os.path.join(model_dir, 'yolov8n.tflite')

print('=' * 60)
print('ML 模型文件检查')
print('=' * 60)
print(f'DeepLabV3 路径: {deeplab_path}')
print(f'  存在: {os.path.exists(deeplab_path)}')
print(f'YOLOv8 路径: {yolo_path}')
print(f'  存在: {os.path.exists(yolo_path)}')

# 检查 TensorFlow
try:
    import tensorflow as tf
    print(f'\n{"=" * 60}')
    print('TensorFlow 检查')
    print('=' * 60)
    print(f'TensorFlow 版本: {tf.__version__}')
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f'GPU 可用: {len(gpus) > 0}')
    if gpus:
        print(f'  GPU 列表: {gpus}')
    
    # 测试 TFLite 解释器
    if os.path.exists(deeplab_path):
        print(f'\n{"=" * 60}')
        print('测试 DeepLabV3 模型')
        print('=' * 60)
        try:
            interpreter = tf.lite.Interpreter(model_path=deeplab_path)
            interpreter.allocate_tensors()
            print('✓ 模型加载成功')
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f'输入形状: {input_details[0]["shape"]}')
            print(f'输入类型: {input_details[0]["dtype"]}')
            print(f'输出形状: {output_details[0]["shape"]}')
            print(f'输出类型: {output_details[0]["dtype"]}')
            
            # 测试推理
            import numpy as np
            input_shape = input_details[0]['shape']
            test_input = np.zeros(input_shape, dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            print(f'✓ 推理测试成功，输出形状: {output.shape}')
            
        except Exception as e:
            print(f'✗ 模型测试失败: {e}')
    else:
        print(f'\n✗ DeepLabV3 模型文件不存在，跳过测试')
    
    if os.path.exists(yolo_path):
        print(f'\n{"=" * 60}')
        print('测试 YOLOv8 模型')
        print('=' * 60)
        try:
            interpreter = tf.lite.Interpreter(model_path=yolo_path)
            interpreter.allocate_tensors()
            print('✓ 模型加载成功')
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f'输入形状: {input_details[0]["shape"]}')
            print(f'输入类型: {input_details[0]["dtype"]}')
            print(f'输出形状: {output_details[0]["shape"]}')
            print(f'输出类型: {output_details[0]["dtype"]}')
            
            # 测试推理
            import numpy as np
            input_shape = input_details[0]['shape']
            test_input = np.zeros(input_shape, dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            print(f'✓ 推理测试成功，输出形状: {output.shape}')
            
        except Exception as e:
            print(f'✗ 模型测试失败: {e}')
    else:
        print(f'\n✗ YOLOv8 模型文件不存在，跳过测试')
        
except ImportError:
    print(f'\n✗ TensorFlow 未安装')
    print('  请运行: pip install tensorflow')
except Exception as e:
    print(f'\n✗ 错误: {e}')
    import traceback
    traceback.print_exc()

# 总结
print(f'\n{"=" * 60}')
print('检查结果总结')
print('=' * 60)

deeplab_exists = os.path.exists(deeplab_path)
yolo_exists = os.path.exists(yolo_path)

if deeplab_exists and yolo_exists:
    print('✓ 所有 ML 模型文件已就位')
    print('✓ ML 方式可以正常运行')
elif deeplab_exists or yolo_exists:
    print('⚠ 部分 ML 模型缺失')
    print('  系统将使用 CV 回退方案补充缺失功能')
else:
    print('✗ 所有 ML 模型文件缺失')
    print('  系统将完全使用 CV 回退方案')
    print('  如需 ML 功能，请运行: python download_models.py')
