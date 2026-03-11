"""
改进的水坑/积水检测模型训练
基于U-Net架构，使用真实场景数据 + 数据增强
"""
import os
import sys
import argparse
import numpy as np
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2

# 添加项目路径
sys.path.insert(0, '.')
from utils.logger import get_logger

logger = get_logger("train_water_model")


class UNet(nn.Module):
    """U-Net for water segmentation"""
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        def up_conv(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        
        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = conv_block(512, 1024)
        
        # Decoder
        self.up4 = up_conv(1024, 512)
        self.dec4 = conv_block(1024, 512)
        self.up3 = up_conv(512, 256)
        self.dec3 = conv_block(512, 256)
        self.up2 = up_conv(256, 128)
        self.dec2 = conv_block(256, 128)
        self.up1 = up_conv(128, 64)
        self.dec1 = conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.sigmoid(self.out(d1))


class WaterDataset(Dataset):
    """水坑检测数据集（含数据增强）"""
    def __init__(self, image_dir, mask_dir, transform=None, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.augment = augment
        
        # 合成数据前缀黑名单（不参与训练，只用真实照片）
        SYNTHETIC_PREFIXES = (
            'synth', 'synthetic', 'aug_',
            'muddy_pu', 'reflecti', 'road_pud', 'sidewalk',
        )
        
        # 获取所有图片，排除合成数据，只保留有掩码的
        all_images = list(self.image_dir.glob("*.jpg")) + \
                     list(self.image_dir.glob("*.png"))
        self.images = [
            p for p in all_images
            if not p.stem.lower().startswith(SYNTHETIC_PREFIXES)
            and (self.mask_dir / (p.stem + '.png')).exists()
        ]
        excluded = len(all_images) - len(self.images)
        print(f"  [数据集] 真实照片 {len(self.images)} 张（已排除合成数据 {excluded} 张）")
    
    def __len__(self):
        return len(self.images)
    
    def _augment(self, image, mask):
        """对图片和掩码同步做数据增强"""
        # 随机水平翻转
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # 随机垂直翻转
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # 随机旋转 ±30度
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        # 随机亮度/对比度/饱和度（只对图片）
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.5, 1.5))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.5, 1.5))
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, random.uniform(0.5, 1.5))
        
        # 随机高斯模糊（模拟不同焦距）
        if random.random() > 0.7:
            from PIL import ImageFilter
            blur_radius = random.uniform(0.5, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return image, mask

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        
        # 加载图片
        image = Image.open(img_path).convert("RGB")
        
        # 加载或生成mask
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", image.size, 0)
        
        # 先统一resize
        image = image.resize((256, 256), Image.BILINEAR)
        mask = mask.resize((256, 256), Image.NEAREST)
        
        # 数据增强（仅训练集）
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # 转为Tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        mask = transforms.ToTensor()(mask)
        
        return image, mask


def train_model(args):
    """训练水坑检测模型"""
    print("=" * 60)
    print("不积洼步 - 改进的水坑检测模型训练")
    print("=" * 60)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    dataset_dir = Path("training_data/water_real")
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    
    if not image_dir.exists() or len(list(image_dir.glob("*.jpg"))) == 0:
        print("\n错误: 没有找到训练数据!")
        print(f"请将水坑图片放入: {image_dir}")
        print(f"请将对应的mask放入: {mask_dir}")
        return
    
    dataset = WaterDataset(image_dir, mask_dir, transform=transform)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(dataset)), [train_size, val_size]
    )
    
    # 训练集启用数据增强，验证集不增强
    train_dataset = WaterDataset(image_dir, mask_dir, transform=transform, augment=True)
    val_dataset   = WaterDataset(image_dir, mask_dir, transform=transform, augment=False)
    
    # 按划分索引取子集
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, list(train_indices))
    val_dataset   = Subset(val_dataset,   list(val_indices))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0)
    
    print(f"\n数据集: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}")
    
    # 创建模型
    model = UNet(in_channels=3, out_channels=1).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    # 训练循环
    best_val_loss = float('inf')
    start_epoch = 0
    checkpoint_dir = Path("training_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 尝试加载已有模型继续训练
    existing_model = checkpoint_dir / "water_model_best.pth"
    if existing_model.exists() and not args.from_scratch:
        print(f"\n发现已有模型: {existing_model}")
        try:
            checkpoint = torch.load(existing_model, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'val_loss' in checkpoint:
                best_val_loss = checkpoint['val_loss']
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            print(f"✓ 已加载模型，从 epoch {start_epoch} 继续训练")
            print(f"  当前最佳验证损失: {best_val_loss:.4f}")
        except Exception as e:
            print(f"⚠ 无法加载已有模型: {e}")
            print("  将从头开始训练")
    
    print("\n开始训练...")
    print("-" * 60)
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                total_epochs = start_epoch + args.epochs
                print(f"\rEpoch [{epoch+1}/{total_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}", end="")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        total_epochs = start_epoch + args.epochs
        print(f"\rEpoch [{epoch+1}/{total_epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f}")
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_dir / "water_model_best.pth")
            print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
    }, checkpoint_dir / "water_model_latest.pth")
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存位置: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train water detection model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--from-scratch", action="store_true", help="从头训练，忽略已有模型")
    
    args = parser.parse_args()
    train_model(args)
