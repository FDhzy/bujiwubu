"""
使用真实照片训练鞋子识别模型
基于UT Zappos50K数据集
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from pathlib import Path
import json
from PIL import Image

print("=" * 70)
print("真实照片鞋子识别模型训练")
print("基于UT Zappos50K数据集")
print("=" * 70)

# 配置
DATASET_DIR = 'training_data/shoe_real'
MODEL_OUTPUT_DIR = 'assets/models/shoe_recognition'
CHECKPOINT_DIR = 'training_checkpoints'

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 5类鞋子
SHOE_CLASSES = ['sandals', 'low_shoes', 'ankle_shoes', 'boots', 'high_boots']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SHOE_CLASSES)}


class RealShoeDataset(Dataset):
    """真实鞋子数据集"""
    
    def __init__(self, image_dir: str, label_dir: str, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # 加载所有样本
        self.samples = []
        image_files = list(Path(image_dir).glob('*.jpg'))
        
        for img_file in image_files:
            label_file = Path(label_dir) / f'{img_file.stem}.txt'
            if label_file.exists():
                with open(label_file) as f:
                    class_id = int(f.read().strip())
                self.samples.append((str(img_file), class_id))
        
        print(f"[数据集] 加载了 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 读取图片
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ShoeResNet50(nn.Module):
    """基于ResNet50的鞋子分类模型"""
    
    def __init__(self, num_classes=5, pretrained=True):
        super(ShoeResNet50, self).__init__()
        
        # 加载预训练ResNet50
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


def train_model(epochs=30, batch_size=32, lr=0.0001):
    """训练模型"""
    
    # 检查数据集
    image_dir = f'{DATASET_DIR}/images'
    label_dir = f'{DATASET_DIR}/labels'
    
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"\n[错误] 数据集不存在: {DATASET_DIR}")
        print("请先运行: python download_shoe_dataset.py")
        return
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    print("\n[1] 准备数据集...")
    full_dataset = RealShoeDataset(image_dir, label_dir, transform=None)
    
    if len(full_dataset) == 0:
        print("[错误] 没有找到有效样本")
        return
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 随机分割
    indices = np.random.permutation(len(full_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 创建子数据集
    class SubsetDataset(Dataset):
        def __init__(self, base_dataset, indices, transform):
            self.base_dataset = base_dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            img_path, label = self.base_dataset.samples[self.indices[idx]]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    train_dataset = SubsetDataset(full_dataset, train_indices, train_transform)
    val_dataset = SubsetDataset(full_dataset, val_indices, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    
    # 创建模型
    print("\n[2] 创建模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    
    model = ShoeResNet50(num_classes=5, pretrained=True).to(device)
    print(f"  总参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # 训练循环
    print("\n[3] 开始训练...")
    print("-" * 70)
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # 学习率调整
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'accuracy': best_val_acc,
                'classes': SHOE_CLASSES
            }
            
            torch.save(checkpoint, f'{CHECKPOINT_DIR}/shoe_real_best.pth')
            torch.save(checkpoint, f'{MODEL_OUTPUT_DIR}/shoe_classifier_real.pth')
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f"\n  ⚠ 早停触发 (连续 {patience} 轮无改善)")
            break
    
    print("-" * 70)
    print(f"  ✓ 训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    
    # 导出模型信息
    model_info = {
        'classes': {i: cls for i, cls in enumerate(SHOE_CLASSES)},
        'model_architecture': 'resnet50',
        'input_size': (224, 224),
        'accuracy': best_val_acc,
        'preprocessing': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'dataset': 'UT Zappos50K'
    }
    
    with open(f'{MODEL_OUTPUT_DIR}/model_info_real.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n[4] 模型已导出到: {MODEL_OUTPUT_DIR}")
    print(f"  - shoe_classifier_real.pth")
    print(f"  - model_info_real.json")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用真实照片训练鞋子识别模型')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("不积洼步 - 真实照片鞋子识别模型训练")
    print("=" * 70)
    print(f"配置: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
