"""
创建障碍物检测 TFLite 模型
使用轻量级目标检测网络
"""
import tensorflow as tf
import numpy as np
import os

print("=" * 60)
print("创建障碍物检测 TFLite 模型")
print("=" * 60)

# 模型参数
INPUT_SIZE = 256
NUM_CLASSES = 5  # 背景、行人、车辆、柱子、其他障碍物
MODEL_PATH = 'assets/models/yolov8_obstacle.tflite'

def create_obstacle_detection_model():
    """创建障碍物检测模型 - 轻量级 MobileNet 风格"""
    
    inputs = tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name='input')
    
    # 特征提取骨干网络 (MobileNetV2 风格)
    # Block 1
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    
    # Block 2 - 深度可分离卷积
    x = tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    x = tf.keras.layers.Conv2D(16, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    
    # Block 3
    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    x = tf.keras.layers.Conv2D(24, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    
    # Block 4
    x = tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    x = tf.keras.layers.Conv2D(24, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    
    # Block 5
    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    x = tf.keras.layers.Conv2D(32, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    
    # Block 6
    x = tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    x = tf.keras.layers.Conv2D(32, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    
    # 特征金字塔网络 (FPN) 风格的多尺度检测
    # 大目标检测头 (8x8)
    fpn_large = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    
    # 上采样并与中层特征融合
    fpn_medium = tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding='same')(fpn_large)
    
    # 检测头输出
    # 每个网格预测: [x, y, w, h, confidence, class_probs...]
    # 使用 8x8 网格，每个网格预测 3 个框
    NUM_ANCHORS = 3
    OUTPUT_PER_ANCHOR = 5 + NUM_CLASSES  # x, y, w, h, conf, classes
    
    # 大目标检测头
    output_large = tf.keras.layers.Conv2D(
        NUM_ANCHORS * OUTPUT_PER_ANCHOR, 
        1, 
        padding='same',
        name='output'
    )(fpn_large)
    
    model = tf.keras.Model(inputs=inputs, outputs=output_large, name='ObstacleDetectionNet')
    return model

# 创建模型
print("\n创建模型架构...")
model = create_obstacle_detection_model()
model.summary()

# 编译模型
print("\n编译模型...")
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

# 创建模拟训练数据
print("\n生成模拟训练数据...")
np.random.seed(42)

X_train = np.random.rand(50, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)

# 模拟目标检测标签 (32x32 网格，每个网格 3 个锚框)
GRID_SIZE = 32  # 256/8 = 32
NUM_ANCHORS = 3
NUM_CLASSES = 5
OUTPUT_PER_ANCHOR = 5 + NUM_CLASSES  # x, y, w, h, conf, classes

Y_train = np.zeros((50, GRID_SIZE, GRID_SIZE, NUM_ANCHORS * OUTPUT_PER_ANCHOR), dtype=np.float32)

# 添加一些模拟目标
for i in range(25):
    # 随机放置目标
    num_objects = np.random.randint(1, 4)
    for _ in range(num_objects):
        grid_y = np.random.randint(0, GRID_SIZE)
        grid_x = np.random.randint(0, GRID_SIZE)
        anchor_idx = np.random.randint(0, NUM_ANCHORS)
        
        # 计算在输出张量中的位置
        offset = anchor_idx * OUTPUT_PER_ANCHOR
        
        # 归一化的边界框坐标 (相对于网格)
        Y_train[i, grid_y, grid_x, offset + 0] = np.random.uniform(0.3, 0.7)  # x center
        Y_train[i, grid_y, grid_x, offset + 1] = np.random.uniform(0.3, 0.7)  # y center
        Y_train[i, grid_y, grid_x, offset + 2] = np.random.uniform(0.2, 0.5)  # width
        Y_train[i, grid_y, grid_x, offset + 3] = np.random.uniform(0.2, 0.5)  # height
        Y_train[i, grid_y, grid_x, offset + 4] = 1.0  # confidence
        
        # 类别 (随机选择一个)
        class_idx = np.random.randint(0, NUM_CLASSES)
        Y_train[i, grid_y, grid_x, offset + 5 + class_idx] = 1.0

# 快速训练
print("\n训练模型（初始化权重）...")
model.fit(
    X_train, Y_train,
    epochs=3,
    batch_size=8,
    verbose=1
)

# 转换为 TFLite
print("\n转换为 TFLite 格式...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 优化设置
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]

# 转换
tflite_model = converter.convert()

# 保存模型
os.makedirs('assets/models', exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"\n✓ 模型已保存: {MODEL_PATH}")
print(f"  大小: {len(tflite_model) / 1024:.1f} KB")

# 验证模型
print("\n验证 TFLite 模型...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✓ 模型验证成功")
print(f"  输入: {input_details[0]['shape']}")
print(f"  输出: {output_details[0]['shape']}")

# 测试推理
print("\n测试推理...")
test_input = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(f"✓ 推理成功，输出形状: {output.shape}")

print("\n" + "=" * 60)
print("障碍物检测模型创建完成!")
print("=" * 60)
