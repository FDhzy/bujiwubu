"""
鞋子识别模型训练系统 - PyTorch ResNet50
专门用于识别不同类型的鞋子：凉鞋、低帮鞋、踝靴、靴子、高筒靴
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Tuple, List
from pathlib import Path
import json

from models.shoe_analyzer import ShoeType
from utils.logger import get_logger

print("=" * 70)
print("鞋子识别模型训练系统 - PyTorch ResNet50")
print("=" * 70)

# 配置
SHOE_MODEL_DIR = 'assets/models/shoe_recognition'
CHECKPOINT_DIR = 'training_checkpoints'
TRAINING_DATA_DIR = 'training_data/shoe'

# 创建目录
os.makedirs(SHOE_MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

# 鞋子类别映射
SHOE_CLASSES = {
    'sandals': 0,      # 凉鞋
    'low_shoes': 1,    # 低帮鞋
    'ankle_shoes': 2,  # 踝靴
    'boots': 3,        # 靴子
    'high_boots': 4,   # 高筒靴/雨靴
}

CLASS_NAMES = {v: k for k, v in SHOE_CLASSES.items()}


class SyntheticShoeDataset(Dataset):
    """合成鞋子数据集生成器"""
    
    def __init__(self, num_samples_per_class: int = 200, transform=None):
        self.num_samples = num_samples_per_class * len(SHOE_CLASSES)
        self.transform = transform
        self.samples = []
        
        # 生成合成数据
        self._generate_synthetic_data(num_samples_per_class)
        
    def _generate_synthetic_data(self, num_per_class: int):
        """生成合成鞋子图像"""
        print(f"\n生成合成训练数据 ({num_per_class} 样本/类别)...")
        
        for class_name, class_idx in SHOE_CLASSES.items():
            for i in range(num_per_class):
                # 生成基础图像
                image = self._create_shoe_image(class_name, i)
                self.samples.append((image, class_idx))
        
        print(f"   ✓ 生成完成: {len(self.samples)} 总样本")
    
    def _create_shoe_image(self, shoe_type: str, seed: int) -> np.ndarray:
        """创建合成鞋子图像"""
        np.random.seed(seed)
        img = np.ones((224, 224, 3), dtype=np.uint8) * 240  # 浅色背景
        
        # 添加随机背景纹理
        noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 根据鞋子类型绘制不同形状
        center_x = 112 + np.random.randint(-20, 20)
        center_y = 140 + np.random.randint(-15, 15)
        
        if shoe_type == 'sandals':
            # 凉鞋：开放结构，带子
            self._draw_sandals(img, center_x, center_y)
        elif shoe_type == 'low_shoes':
            # 低帮鞋：低帮封闭结构
            self._draw_low_shoes(img, center_x, center_y)
        elif shoe_type == 'ankle_shoes':
            # 踝靴：中等高度
            self._draw_ankle_shoes(img, center_x, center_y)
        elif shoe_type == 'boots':
            # 靴子：高帮
            self._draw_boots(img, center_x, center_y)
        elif shoe_type == 'high_boots':
            # 高筒靴：最高
            self._draw_high_boots(img, center_x, center_y)
        
        # 添加随机光照变化
        img = self._apply_lighting_variation(img)
        
        return img
    
    def _draw_sandals(self, img: np.ndarray, cx: int, cy: int):
        """绘制凉鞋"""
        color = (np.random.randint(100, 200), np.random.randint(80, 160), np.random.randint(60, 140))
        # 鞋底
        cv2.ellipse(img, (cx, cy), (50, 25), 0, 0, 360, color, -1)
        # 带子
        cv2.line(img, (cx-30, cy-10), (cx+30, cy-10), (color[0]+30, color[1]+30, color[2]+30), 8)
        cv2.line(img, (cx-20, cy-25), (cx+20, cy-25), (color[0]+20, color[1]+20, color[2]+20), 6)
    
    def _draw_low_shoes(self, img: np.ndarray, cx: int, cy: int):
        """绘制低帮鞋"""
        color = (np.random.randint(60, 150), np.random.randint(60, 150), np.random.randint(60, 150))
        # 鞋底
        cv2.ellipse(img, (cx, cy+5), (55, 28), 0, 0, 360, (40, 40, 40), -1)
        # 鞋面
        cv2.ellipse(img, (cx, cy-10), (52, 30), 0, 0, 180, color, -1)
        # 鞋口
        cv2.ellipse(img, (cx, cy-15), (25, 12), 0, 0, 360, (240, 240, 240), -1)
    
    def _draw_ankle_shoes(self, img: np.ndarray, cx: int, cy: int):
        """绘制踝靴"""
        color = (np.random.randint(80, 160), np.random.randint(60, 140), np.random.randint(40, 120))
        # 鞋底
        cv2.ellipse(img, (cx, cy+5), (55, 28), 0, 0, 360, (40, 40, 40), -1)
        # 鞋面
        cv2.ellipse(img, (cx, cy-5), (52, 35), 0, 0, 180, color, -1)
        # 鞋帮（到脚踝）
        cv2.rectangle(img, (cx-30, cy-50), (cx+30, cy-15), color, -1)
        cv2.ellipse(img, (cx, cy-50), (30, 10), 0, 0, 360, (color[0]-20, color[1]-20, color[2]-20), -1)
    
    def _draw_boots(self, img: np.ndarray, cx: int, cy: int):
        """绘制靴子"""
        color = (np.random.randint(60, 140), np.random.randint(40, 120), np.random.randint(30, 100))
        # 鞋底
        cv2.ellipse(img, (cx, cy+5), (55, 28), 0, 0, 360, (40, 40, 40), -1)
        # 鞋面
        cv2.ellipse(img, (cx, cy), (52, 32), 0, 0, 180, color, -1)
        # 高鞋帮
        cv2.rectangle(img, (cx-32, cy-90), (cx+32, cy-10), color, -1)
        cv2.ellipse(img, (cx, cy-90), (32, 12), 0, 0, 360, (color[0]-20, color[1]-20, color[2]-20), -1)
    
    def _draw_high_boots(self, img: np.ndarray, cx: int, cy: int):
        """绘制高筒靴/雨靴"""
        color = (np.random.randint(50, 120), np.random.randint(80, 150), np.random.randint(100, 180))
        # 鞋底
        cv2.ellipse(img, (cx, cy+5), (58, 30), 0, 0, 360, (40, 40, 40), -1)
        # 鞋面
        cv2.ellipse(img, (cx, cy), (55, 35), 0, 0, 180, color, -1)
        # 非常高鞋帮（到小腿）
        cv2.rectangle(img, (cx-35, cy-130), (cx+35, cy-10), color, -1)
        cv2.ellipse(img, (cx, cy-130), (35, 15), 0, 0, 360, (color[0]-30, color[1]-30, color[2]-30), -1)
    
    def _apply_lighting_variation(self, img: np.ndarray) -> np.ndarray:
        """应用光照变化"""
        # 随机亮度调整
        brightness = np.random.uniform(0.8, 1.2)
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)
        
        # 随机对比度
        contrast = np.random.uniform(0.9, 1.1)
        mean = np.mean(img)
        img = np.clip((img - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        return img
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.samples[idx]
        
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = transforms.ToPILImage()(image)
        
        if self.transform:
            pil_image = self.transform(pil_image)
        
        return pil_image, label


class ShoeRecognitionModel(nn.Module):
    """基于 ResNet50 的鞋子识别模型"""
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(ShoeRecognitionModel, self).__init__()
        
        # 加载预训练的 ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # 替换最后的全连接层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


class ShoeModelTrainer:
    """鞋子识别模型训练器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = get_logger("shoe_trainer")
        self.model = None
        self.best_accuracy = 0.0
        
        print(f"\n使用设备: {self.device}")
        
        # 数据增强和预处理
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def prepare_data(self, samples_per_class: int = 200):
        """准备训练数据"""
        print("\n[1] 准备训练数据...")
        
        # 创建训练集和验证集
        full_dataset = SyntheticShoeDataset(
            num_samples_per_class=samples_per_class,
            transform=None  # 将在 __getitem__ 中应用
        )
        
        # 分割训练集和验证集 (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # 包装数据集以应用不同的变换
        train_dataset = TransformedDataset(train_dataset, self.train_transform)
        val_dataset = TransformedDataset(val_dataset, self.val_transform)
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=0
        )
        
        print(f"   ✓ 训练集: {len(train_dataset)} 样本")
        print(f"   ✓ 验证集: {len(val_dataset)} 样本")
    
    def build_model(self):
        """构建模型"""
        print("\n[2] 构建模型...")
        
        self.model = ShoeRecognitionModel(num_classes=5, pretrained=True)
        self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        print("   ✓ 模型构建完成")
        print(f"   总参数: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个 epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc
    
    def train(self, epochs: int = 30):
        """训练模型"""
        print(f"\n[3] 开始训练 (最大 {epochs} 轮)...")
        print("-" * 70)
        
        best_val_acc = 0.0
        patience = 7
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # 学习率调整
            self.scheduler.step(val_acc)
            
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_accuracy = val_acc
                self.save_checkpoint(is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f"\n   ⚠ 早停触发 (连续 {patience} 轮无改善)")
                break
        
        print("-" * 70)
        print(f"   ✓ 训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    
    def save_checkpoint(self, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': 0,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'accuracy': self.best_accuracy,
            'classes': SHOE_CLASSES
        }
        
        # 保存最新检查点
        latest_path = os.path.join(CHECKPOINT_DIR, 'shoe_model_latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(CHECKPOINT_DIR, 'shoe_model_best.pth')
            torch.save(checkpoint, best_path)
            
            # 同时保存为应用使用的格式
            model_path = os.path.join(SHOE_MODEL_DIR, 'shoe_classifier.pth')
            torch.save(checkpoint, model_path)
    
    def export_model(self):
        """导出最终模型"""
        print("\n[4] 导出模型...")
        
        # 保存完整模型信息
        model_info = {
            'classes': SHOE_CLASSES,
            'class_names': CLASS_NAMES,
            'model_architecture': 'resnet50',
            'input_size': (224, 224),
            'accuracy': self.best_accuracy,
            'preprocessing': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
        
        info_path = os.path.join(SHOE_MODEL_DIR, 'model_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ 模型已导出到: {SHOE_MODEL_DIR}")
        print(f"   ✓ 模型信息: {info_path}")
        print(f"   ✓ 验证准确率: {self.best_accuracy:.2f}%")


class TransformedDataset(Dataset):
    """包装数据集以应用变换"""
    
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # 获取原始图像和标签
        image, label = self.base_dataset[idx]
        
        # 如果 image 已经是 tensor，说明原始数据集已经应用了变换
        if isinstance(image, torch.Tensor):
            return image, label
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练鞋子识别模型')
    parser.add_argument('--samples', type=int, default=200, 
                        help='每类样本数 (默认: 200)')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='训练轮数 (默认: 30)')
    parser.add_argument('--skip-training', action='store_true',
                        help='跳过训练，仅导出当前最佳模型')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("不积洼步 - 鞋子识别模型训练")
    print("=" * 70)
    print(f"配置: 每类 {args.samples} 样本, 最大 {args.epochs} 轮")
    
    trainer = ShoeModelTrainer()
    
    if not args.skip_training:
        trainer.prepare_data(samples_per_class=args.samples)
        trainer.build_model()
        trainer.train(epochs=args.epochs)
    
    trainer.export_model()
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"\n模型文件位置:")
    print(f"  - 检查点: {CHECKPOINT_DIR}/shoe_model_best.pth")
    print(f"  - 应用模型: {SHOE_MODEL_DIR}/shoe_classifier.pth")
    print(f"  - 模型信息: {SHOE_MODEL_DIR}/model_info.json")
    print("\n提示:")
    print("  1. 模型已针对 5 种鞋子类型优化")
    print("  2. 使用合成数据训练，真实场景可能需要微调")
    print("  3. 要进一步提高精度，可收集真实鞋子图片进行迁移学习")
    print("=" * 70)


if __name__ == '__main__':
    main()
