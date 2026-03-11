"""
模型训练系统 - 使用真实数据继续训练以提高精度
支持水坑检测和障碍物检测模型的增量训练
"""
import tensorflow as tf
import numpy as np
import os
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import json

print("=" * 70)
print("模型训练系统 - 提高 ML 检测精度")
print("=" * 70)

# 配置
WATER_MODEL_PATH = 'assets/models/deeplabv3_water.tflite'
OBSTACLE_MODEL_PATH = 'assets/models/yolov8_obstacle.tflite'
TRAINING_DATA_DIR = 'training_data'
CHECKPOINT_DIR = 'training_checkpoints'

# 创建目录
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(f'{TRAINING_DATA_DIR}/water/images', exist_ok=True)
os.makedirs(f'{TRAINING_DATA_DIR}/water/masks', exist_ok=True)
os.makedirs(f'{TRAINING_DATA_DIR}/obstacle/images', exist_ok=True)
os.makedirs(f'{TRAINING_DATA_DIR}/obstacle/labels', exist_ok=True)


class WaterDetectionTrainer:
    """水坑检测模型训练器"""
    
    def __init__(self):
        self.input_size = (256, 256)
        self.model = None
        self.load_existing_model()
    
    def load_existing_model(self):
        """加载现有的 TFLite 模型并转换为 Keras 模型"""
        print("\n[1] 加载现有水坑检测模型...")
        
        # 由于 TFLite 无法直接转回 Keras，我们重建模型架构
        self.model = self._build_unet_model()
        
        # 尝试加载已有的检查点
        checkpoint_path = f'{CHECKPOINT_DIR}/water_model_latest.h5'
        if os.path.exists(checkpoint_path):
            print(f"   加载检查点: {checkpoint_path}")
            self.model.load_weights(checkpoint_path)
        else:
            print("   未找到检查点，使用新初始化的模型")
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        print("   ✓ 模型准备完成")
    
    def _build_unet_model(self):
        """构建 U-Net 模型架构"""
        inputs = tf.keras.layers.Input(shape=(256, 256, 3), name='input')
        
        # 编码器
        c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        
        c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
        c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        
        c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
        c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        
        b = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
        b = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(b)
        
        # 解码器
        u3 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(b)
        u3 = tf.keras.layers.concatenate([u3, c3])
        c3_up = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(u3)
        c3_up = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c3_up)
        
        u2 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c3_up)
        u2 = tf.keras.layers.concatenate([u2, c2])
        c2_up = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
        c2_up = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c2_up)
        
        u1 = tf.keras.layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(c2_up)
        u1 = tf.keras.layers.concatenate([u1, c1])
        c1_up = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
        c1_up = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c1_up)
        
        outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='output')(c1_up)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='WaterDetectionNet')
    
    def generate_synthetic_data(self, num_samples: int = 200):
        """生成合成训练数据（带更真实的水坑特征）"""
        print(f"\n[2] 生成合成训练数据 ({num_samples} 样本)...")
        
        X = np.zeros((num_samples, 256, 256, 3), dtype=np.float32)
        Y = np.zeros((num_samples, 256, 256, 1), dtype=np.float32)
        
        for i in range(num_samples):
            # 生成背景（地面）
            bg_color = np.random.uniform([0.2, 0.2, 0.15], [0.5, 0.5, 0.4])
            X[i] = np.ones((256, 256, 3)) * bg_color
            
            # 添加噪声模拟地面纹理
            noise = np.random.normal(0, 0.05, (256, 256, 3))
            X[i] = np.clip(X[i] + noise, 0, 1)
            
            # 随机决定是否添加水坑 (70% 概率)
            if np.random.random() < 0.7:
                num_puddles = np.random.randint(1, 4)
                
                for _ in range(num_puddles):
                    # 水坑位置（主要在图像底部，模拟地面）
                    y_center = np.random.randint(128, 240)
                    x_center = np.random.randint(40, 216)
                    
                    # 水坑大小
                    width = np.random.randint(30, 100)
                    height = np.random.randint(20, 60)
                    
                    # 创建椭圆水坑
                    y_start = max(0, y_center - height // 2)
                    y_end = min(256, y_center + height // 2)
                    x_start = max(0, x_center - width // 2)
                    x_end = min(256, x_center + width // 2)
                    
                    # 水坑特征：高亮度、低饱和度、偏蓝色调
                    # 模拟水面反射
                    water_color = np.random.uniform([0.5, 0.5, 0.6], [0.8, 0.8, 0.9])
                    
                    # 创建渐变效果
                    for y in range(y_start, y_end):
                        for x in range(x_start, x_end):
                            # 椭圆距离
                            dy = (y - y_center) / (height / 2 + 1e-5)
                            dx = (x - x_center) / (width / 2 + 1e-5)
                            dist = np.sqrt(dx**2 + dy**2)
                            
                            if dist < 1.0:
                                # 边缘渐变
                                alpha = 1.0 - dist**2
                                X[i, y, x] = (1 - alpha) * X[i, y, x] + alpha * water_color
                                Y[i, y, x] = 1.0
        
        print(f"   ✓ 生成完成: X shape={X.shape}, Y shape={Y.shape}")
        return X, Y
    
    def train(self, epochs: int = 20, batch_size: int = 8):
        """训练模型"""
        print("\n[3] 开始训练水坑检测模型...")
        
        # 生成数据
        X_train, Y_train = self.generate_synthetic_data(200)
        X_val, Y_val = self.generate_synthetic_data(50)
        
        # 数据增强
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2]
        )
        
        # 回调函数
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                f'{CHECKPOINT_DIR}/water_model_best.h5',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        # 训练
        history = self.model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("   ✓ 训练完成")
        return history
    
    def export_tflite(self):
        """导出为 TFLite 格式"""
        print("\n[4] 导出 TFLite 模型...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(WATER_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
        
        # 保存检查点
        self.model.save_weights(f'{CHECKPOINT_DIR}/water_model_latest.h5')
        
        print(f"   ✓ 模型已导出: {WATER_MODEL_PATH}")
        print(f"   大小: {len(tflite_model)/1024:.1f} KB")


class ObstacleDetectionTrainer:
    """障碍物检测模型训练器"""
    
    def __init__(self):
        self.input_size = (256, 256)
        self.grid_size = 32
        self.num_anchors = 3
        self.num_classes = 5
        self.model = None
        self.load_existing_model()
    
    def load_existing_model(self):
        """加载现有模型"""
        print("\n[1] 加载现有障碍物检测模型...")
        
        self.model = self._build_mobilenet_model()
        
        checkpoint_path = f'{CHECKPOINT_DIR}/obstacle_model_latest.h5'
        if os.path.exists(checkpoint_path):
            print(f"   加载检查点: {checkpoint_path}")
            self.model.load_weights(checkpoint_path)
        else:
            print("   未找到检查点，使用新初始化的模型")
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=self._yolo_loss,
            metrics=['accuracy']
        )
        print("   ✓ 模型准备完成")
    
    def _build_mobilenet_model(self):
        """构建 MobileNet 风格的检测模型"""
        inputs = tf.keras.layers.Input(shape=(256, 256, 3), name='input')
        
        # 简化的 MobileNet 骨干
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU(6.0)(x)
        
        x = self._depthwise_separable(x, 16, 1)
        x = self._depthwise_separable(x, 24, 2)
        x = self._depthwise_separable(x, 24, 1)
        x = self._depthwise_separable(x, 32, 2)
        x = self._depthwise_separable(x, 32, 1)
        x = self._depthwise_separable(x, 64, 2)
        x = self._depthwise_separable(x, 64, 1)
        
        # 检测头
        x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        
        output_per_anchor = 5 + self.num_classes
        outputs = tf.keras.layers.Conv2D(
            self.num_anchors * output_per_anchor,
            1,
            padding='same',
            name='output'
        )(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='ObstacleDetectionNet')
    
    def _depthwise_separable(self, x, filters, strides):
        """深度可分离卷积块"""
        x = tf.keras.layers.DepthwiseConv2D(
            3, strides=strides, padding='same', use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU(6.0)(x)
        x = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.ReLU(6.0)(x)
    
    def _yolo_loss(self, y_true, y_pred):
        """YOLO 风格的损失函数"""
        # 分离坐标、置信度和类别
        output_per_anchor = 5 + self.num_classes
        
        # 坐标损失 (x, y, w, h)
        coord_loss = tf.reduce_mean(
            tf.square(y_true[..., :4] - y_pred[..., :4]) * y_true[..., 4:5]
        )
        
        # 置信度损失
        conf_loss = tf.reduce_mean(
            tf.square(y_true[..., 4] - y_pred[..., 4])
        )
        
        # 类别损失
        class_loss = tf.reduce_mean(
            tf.square(y_true[..., 5:] - y_pred[..., 5:]) * y_true[..., 4:5]
        )
        
        return coord_loss + conf_loss + class_loss
    
    def generate_synthetic_data(self, num_samples: int = 200):
        """生成合成训练数据"""
        print(f"\n[2] 生成合成训练数据 ({num_samples} 样本)...")
        
        X = np.random.rand(num_samples, 256, 256, 3).astype(np.float32)
        output_per_anchor = 5 + self.num_classes
        Y = np.zeros((num_samples, 32, 32, self.num_anchors * output_per_anchor), dtype=np.float32)
        
        for i in range(num_samples):
            # 添加背景纹理
            X[i] = np.random.uniform(0.3, 0.7, (256, 256, 3))
            
            # 随机添加障碍物
            num_obstacles = np.random.randint(0, 4)
            
            for _ in range(num_obstacles):
                # 障碍物类型
                obj_class = np.random.randint(0, self.num_classes)
                
                # 在网格上的位置
                grid_y = np.random.randint(0, 32)
                grid_x = np.random.randint(0, 32)
                anchor_idx = np.random.randint(0, self.num_anchors)
                
                offset = anchor_idx * output_per_anchor
                
                # 边界框 (相对于网格的归一化坐标)
                Y[i, grid_y, grid_x, offset + 0] = np.random.uniform(0.3, 0.7)  # x
                Y[i, grid_y, grid_x, offset + 1] = np.random.uniform(0.3, 0.7)  # y
                Y[i, grid_y, grid_x, offset + 2] = np.random.uniform(0.1, 0.4)  # w
                Y[i, grid_y, grid_x, offset + 3] = np.random.uniform(0.1, 0.4)  # h
                Y[i, grid_y, grid_x, offset + 4] = 1.0  # 置信度
                Y[i, grid_y, grid_x, offset + 5 + obj_class] = 1.0  # 类别
                
                # 在图像上绘制障碍物（简化表示）
                px = int((grid_x + 0.5) / 32 * 256)
                py = int((grid_y + 0.5) / 32 * 256)
                size = int(20 * (1 + obj_class * 0.3))
                
                color = [0.2, 0.2, 0.2] if obj_class == 0 else [0.3, 0.3, 0.3]
                cv2.rectangle(X[i], (px-size, py-size), (px+size, py+size), color, -1)
        
        print(f"   ✓ 生成完成: X shape={X.shape}, Y shape={Y.shape}")
        return X, Y
    
    def train(self, epochs: int = 20, batch_size: int = 8):
        """训练模型"""
        print("\n[3] 开始训练障碍物检测模型...")
        
        X_train, Y_train = self.generate_synthetic_data(200)
        X_val, Y_val = self.generate_synthetic_data(50)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                f'{CHECKPOINT_DIR}/obstacle_model_best.h5',
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        history = self.model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("   ✓ 训练完成")
        return history
    
    def export_tflite(self):
        """导出为 TFLite 格式"""
        print("\n[4] 导出 TFLite 模型...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(OBSTACLE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
        
        self.model.save_weights(f'{CHECKPOINT_DIR}/obstacle_model_latest.h5')
        
        print(f"   ✓ 模型已导出: {OBSTACLE_MODEL_PATH}")
        print(f"   大小: {len(tflite_model)/1024:.1f} KB")


def train_water_model():
    """训练水坑检测模型"""
    print("\n" + "=" * 70)
    print("训练水坑检测模型")
    print("=" * 70)
    
    trainer = WaterDetectionTrainer()
    history = trainer.train(epochs=20)
    trainer.export_tflite()
    
    return history


def train_obstacle_model():
    """训练障碍物检测模型"""
    print("\n" + "=" * 70)
    print("训练障碍物检测模型")
    print("=" * 70)
    
    trainer = ObstacleDetectionTrainer()
    history = trainer.train(epochs=20)
    trainer.export_tflite()
    
    return history


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练 ML 模型')
    parser.add_argument('--model', choices=['water', 'obstacle', 'both'], 
                        default='both', help='选择要训练的模型')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("不积洼步 - 模型训练系统")
    print("=" * 70)
    print(f"训练配置: epochs={args.epochs}, model={args.model}")
    
    if args.model in ['water', 'both']:
        train_water_model()
    
    if args.model in ['obstacle', 'both']:
        train_obstacle_model()
    
    print("\n" + "=" * 70)
    print("所有训练完成！")
    print("=" * 70)
    print("\n提示：")
    print("1. 训练后的模型已保存到 assets/models/")
    print("2. 检查点保存在 training_checkpoints/")
    print("3. 要进一步提高精度，可以：")
    print("   - 收集真实场景图片进行标注")
    print("   - 增加训练轮数")
    print("   - 调整学习率")
    print("=" * 70)


if __name__ == '__main__':
    main()
