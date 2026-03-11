"""
简化版鞋子识别模型训练 - 使用轻量级CNN
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import json

print("=" * 60)
print("简化版鞋子识别模型训练")
print("=" * 60)

# 配置
SHOE_MODEL_DIR = 'assets/models/shoe_recognition'
CHECKPOINT_DIR = 'training_checkpoints'
os.makedirs(SHOE_MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SHOE_CLASSES = {
    'sandals': 0,
    'low_shoes': 1,
    'ankle_shoes': 2,
    'boots': 3,
    'high_boots': 4,
}
CLASS_NAMES = {v: k for k, v in SHOE_CLASSES.items()}


def create_shoe_image(shoe_type, seed):
    """创建合成鞋子图像"""
    np.random.seed(seed)
    img = np.ones((224, 224, 3), dtype=np.uint8) * 240
    noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    cx, cy = 112 + np.random.randint(-20, 20), 140 + np.random.randint(-15, 15)
    
    if shoe_type == 'sandals':
        color = (np.random.randint(100, 200), np.random.randint(80, 160), np.random.randint(60, 140))
        cv2.ellipse(img, (cx, cy), (50, 25), 0, 0, 360, color, -1)
        cv2.line(img, (cx-30, cy-10), (cx+30, cy-10), (color[0]+30, color[1]+30, color[2]+30), 8)
    elif shoe_type == 'low_shoes':
        color = (np.random.randint(60, 150), np.random.randint(60, 150), np.random.randint(60, 150))
        cv2.ellipse(img, (cx, cy+5), (55, 28), 0, 0, 360, (40, 40, 40), -1)
        cv2.ellipse(img, (cx, cy-10), (52, 30), 0, 0, 180, color, -1)
    elif shoe_type == 'ankle_shoes':
        color = (np.random.randint(80, 160), np.random.randint(60, 140), np.random.randint(40, 120))
        cv2.ellipse(img, (cx, cy+5), (55, 28), 0, 0, 360, (40, 40, 40), -1)
        cv2.rectangle(img, (cx-30, cy-50), (cx+30, cy-15), color, -1)
    elif shoe_type == 'boots':
        color = (np.random.randint(60, 140), np.random.randint(40, 120), np.random.randint(30, 100))
        cv2.ellipse(img, (cx, cy+5), (55, 28), 0, 0, 360, (40, 40, 40), -1)
        cv2.rectangle(img, (cx-32, cy-90), (cx+32, cy-10), color, -1)
    elif shoe_type == 'high_boots':
        color = (np.random.randint(50, 120), np.random.randint(80, 150), np.random.randint(100, 180))
        cv2.ellipse(img, (cx, cy+5), (58, 30), 0, 0, 360, (40, 40, 40), -1)
        cv2.rectangle(img, (cx-35, cy-130), (cx+35, cy-10), color, -1)
    
    brightness = np.random.uniform(0.8, 1.2)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    return img


class SimpleShoeDataset(Dataset):
    """简单鞋子数据集"""
    def __init__(self, num_per_class=50, transform=None):
        self.samples = []
        self.transform = transform
        
        print(f"生成 {num_per_class} 样本/类别...")
        for class_name, class_idx in SHOE_CLASSES.items():
            for i in range(num_per_class):
                img = create_shoe_image(class_name, i + class_idx * 1000)
                self.samples.append((img, class_idx))
        print(f"  ✓ 总样本: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image, label = self.samples[idx]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = transforms.ToPILImage()(image)
        if self.transform:
            pil_image = self.transform(pil_image)
        return pil_image, label


class SimpleShoeNet(nn.Module):
    """轻量级鞋子识别网络"""
    def __init__(self, num_classes=5):
        super(SimpleShoeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_model():
    """训练模型"""
    device = torch.device('cpu')
    print(f"\n使用设备: {device}")
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    print("\n[1] 准备数据...")
    full_dataset = SimpleShoeDataset(num_per_class=50, transform=None)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, len(full_dataset)))
    
    # 手动创建训练/验证集
    train_samples = [full_dataset[i] for i in train_indices]
    val_samples = [full_dataset[i] for i in val_indices]
    
    class SimpleDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            pil_img, label = self.samples[idx]
            # pil_img is already a PIL Image from full_dataset
            if self.transform:
                pil_img = self.transform(pil_img)
            return pil_img, label
    
    train_dataset = SimpleDataset(train_samples, train_transform)
    val_dataset = SimpleDataset(val_samples, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"   训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # 创建模型
    print("\n[2] 构建模型...")
    model = SimpleShoeNet(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"   总参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    print("\n[3] 开始训练...")
    print("-" * 60)
    best_acc = 0.0
    
    for epoch in range(10):
        # 训练
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # 验证
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1:2d}/10 | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存最佳模型
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'accuracy': best_acc,
                'classes': SHOE_CLASSES
            }
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'shoe_model_best.pth'))
            torch.save(checkpoint, os.path.join(SHOE_MODEL_DIR, 'shoe_classifier.pth'))
    
    print("-" * 60)
    print(f"   ✓ 训练完成! 最佳准确率: {best_acc:.1f}%")
    
    # 导出模型信息
    model_info = {
        'classes': SHOE_CLASSES,
        'class_names': CLASS_NAMES,
        'model_architecture': 'SimpleShoeNet',
        'input_size': (224, 224),
        'accuracy': best_acc,
        'preprocessing': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    with open(os.path.join(SHOE_MODEL_DIR, 'model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n[4] 模型已导出到: {SHOE_MODEL_DIR}")
    print(f"   - shoe_classifier.pth")
    print(f"   - model_info.json")
    
    return model


if __name__ == '__main__':
    train_model()
