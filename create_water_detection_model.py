"""
创建水坑检测 TFLite 模型
使用轻量级卷积神经网络进行水坑语义分割
"""
import tensorflow as tf
import numpy as np
import os

print("=" * 60)
print("创建水坑检测 TFLite 模型")
print("=" * 60)

# 模型参数
INPUT_SIZE = 256
MODEL_PATH = 'assets/models/deeplabv3_water.tflite'

# 创建轻量级 U-Net 风格的模型
def create_water_detection_model():
    """创建水坑检测模型 - 轻量级 U-Net"""
    
    # 输入层
    inputs = tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name='input')
    
    # 编码器 (下采样)
    # Block 1
    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    
    # Block 2
    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    # Block 3
    c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
    # Bridge
    b = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    b = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(b)
    
    # 解码器 (上采样)
    # Block 3 Up
    u3 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(b)
    u3 = tf.keras.layers.concatenate([u3, c3])
    c3_up = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(u3)
    c3_up = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c3_up)
    
    # Block 2 Up
    u2 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c3_up)
    u2 = tf.keras.layers.concatenate([u2, c2])
    c2_up = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    c2_up = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c2_up)
    
    # Block 1 Up
    u1 = tf.keras.layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(c2_up)
    u1 = tf.keras.layers.concatenate([u1, c1])
    c1_up = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    c1_up = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c1_up)
    
    # 输出层 - 水坑分割掩码
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='output')(c1_up)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='WaterDetectionNet')
    return model

# 创建模型
print("\n创建模型架构...")
model = create_water_detection_model()
model.summary()

# 编译模型
print("\n编译模型...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 创建模拟训练数据（用于初始化模型权重）
print("\n生成模拟训练数据...")
np.random.seed(42)

# 模拟数据：包含水坑特征的图像
# 水坑特征：低饱和度、高亮度、特定纹理
X_train = np.random.rand(100, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)
Y_train = np.zeros((100, INPUT_SIZE, INPUT_SIZE, 1), dtype=np.float32)

# 添加一些模拟的水坑样本（底部区域的高亮度低饱和度区域）
for i in range(50):
    # 在图像底部创建水坑区域
    y_start = np.random.randint(INPUT_SIZE // 2, INPUT_SIZE - 50)
    x_start = np.random.randint(0, INPUT_SIZE - 100)
    width = np.random.randint(50, 100)
    height = np.random.randint(30, 60)
    
    # 水坑特征：高亮度、低饱和度
    X_train[i, y_start:y_start+height, x_start:x_start+width, :] = [
        0.7, 0.7, 0.8  # BGR 高亮度
    ]
    Y_train[i, y_start:y_start+height, x_start:x_start+width, :] = 1.0

# 快速训练几轮以初始化权重
print("\n训练模型（初始化权重）...")
model.fit(
    X_train, Y_train,
    epochs=5,
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
print("水坑检测模型创建完成!")
print("=" * 60)
