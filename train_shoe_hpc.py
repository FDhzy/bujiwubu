"""
鞋型识别模型 - 复旦并行智算云 HPC 训练脚本
基于 UT Zappos50K 数据集，ResNet50，支持多 GPU

上传文件清单：
  train_shoe_hpc.py          ← 本脚本
  submit_shoe_hpc.slurm      ← SLURM 作业文件
  training_data/shoe_real/   ← 数据集（images/ + labels/）

训练输出：
  checkpoints/shoe_best.pth  ← 最佳权重
  checkpoints/shoe_last.pth  ← 最新权重
"""

import os
import argparse
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="鞋型识别 ResNet50 训练")
    parser.add_argument("--data-dir",    default="training_data/shoe_real", help="数据集根目录")
    parser.add_argument("--output-dir",  default="checkpoints",             help="权重输出目录")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=64,   help="单 GPU batch size")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--workers",     type=int,   default=4,    help="DataLoader 工作进程数")
    parser.add_argument("--resume",      default="",               help="从 checkpoint 继续训练")
    parser.add_argument("--num-classes", type=int,   default=3,    help="实际类别数（数据中有3类）")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 数据集
# ─────────────────────────────────────────────────────────────────────────────
# UT Zappos50K 文件夹名 → 项目鞋型类别映射
# 目录结构: ut-zap50k-images-square/{Boots,Sandals,Shoes,Slippers}/{品牌}/图片.jpg
FOLDER_TO_LABEL = {
    'Sandals':  0,   # sandals
    'Slippers': 0,   # sandals（拖鞋归入凉鞋）
    'Shoes':    1,   # low_shoes
    'Boots':    2,   # boots
}
LABEL_NAMES = ['sandals', 'low_shoes', 'boots']


class ShoeDataset(Dataset):
    """从 ut-zap50k-images-square/{类别文件夹} 目录结构读取数据"""

    def __init__(self, data_dir: str, transform=None):
        self.transform = transform
        self.samples   = []

        # 支持两种路径：data_dir 本身 或 data_dir/images/ut-zap50k-images-square
        candidates = [
            Path(data_dir) / 'images' / 'ut-zap50k-images-square',
            Path(data_dir) / 'ut-zap50k-images-square',
            Path(data_dir),
        ]
        root = None
        for c in candidates:
            if c.exists() and any(c.iterdir()):
                # 确认有已知类别子目录
                if any((c / k).exists() for k in FOLDER_TO_LABEL):
                    root = c
                    break

        if root is None:
            print(f"[Dataset] 警告：找不到 ut-zap50k-images-square 目录，尝试路径:")
            for c in candidates:
                print(f"          {c}  exists={c.exists()}")
        else:
            print(f"[Dataset] 数据根目录: {root}")
            for folder_name, label in FOLDER_TO_LABEL.items():
                folder = root / folder_name
                if not folder.exists():
                    continue
                # 递归查找所有 jpg
                imgs = list(folder.rglob('*.jpg'))
                for img_path in imgs:
                    self.samples.append((str(img_path), label))

        print(f"[Dataset] 共 {len(self.samples)} 个样本")
        dist_counter = Counter(s[1] for s in self.samples)
        for idx, name in enumerate(LABEL_NAMES):
            print(f"         类别 {idx} {name}: {dist_counter.get(idx, 0)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ─────────────────────────────────────────────────────────────────────────────
# 模型
# ─────────────────────────────────────────────────────────────────────────────
class ShoeResNet50(nn.Module):
    """ResNet50 + 自定义分类头（与 pytorch_shoe_model.py 兼容）"""

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)


# ─────────────────────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    args = parse_args()

    # ── 输出目录 ──────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 设备 ──────────────────────────────────────────────
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        device = torch.device("cuda")
        print(f"[Device] CUDA 可用，GPU 数量: {n_gpu}")
        for i in range(n_gpu):
            print(f"         GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device("cpu")
        print("[Device] 使用 CPU（训练会很慢）")

    # ── 数据增强 ──────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── 数据集 ────────────────────────────────────────────
    full_ds = ShoeDataset(args.data_dir, transform=None)
    if len(full_ds) == 0:
        print("[错误] 数据集为空，请检查路径")
        return

    n = len(full_ds)
    indices = np.random.permutation(n)
    split   = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    # 临时包装以切换 transform
    class _Subset(Dataset):
        def __init__(self, base, idxs, tf):
            self.base = base; self.idxs = idxs; self.tf = tf
        def __len__(self): return len(self.idxs)
        def __getitem__(self, i):
            path, label = self.base.samples[self.idxs[i]]
            img = Image.open(path).convert("RGB")
            return self.tf(img), label

    train_ds = _Subset(full_ds, train_idx, train_tf)
    val_ds   = _Subset(full_ds, val_idx,   val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True)

    print(f"[Split] 训练: {len(train_ds)}  验证: {len(val_ds)}")

    # ── 模型 ──────────────────────────────────────────────
    model = ShoeResNet50(num_classes=args.num_classes, pretrained=True)
    if torch.cuda.device_count() > 1:
        print(f"[Model] 使用 DataParallel，{torch.cuda.device_count()} 个 GPU")
        model = nn.DataParallel(model)
    model = model.to(device)

    # ── 损失 / 优化器 / 调度 ──────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── 续训 ──────────────────────────────────────────────
    start_epoch   = 0
    best_val_acc  = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"[Resume] 从 epoch {start_epoch} 继续，最佳验证准确率 {best_val_acc:.2f}%")

    # ── 训练循环 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"开始训练: epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print("=" * 70)

    patience_counter = 0
    patience         = 10

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device, train=False)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train {train_acc:.2f}% (loss={train_loss:.4f}) | "
              f"Val {val_acc:.2f}% (loss={val_loss:.4f}) | "
              f"{elapsed:.0f}s | LR={scheduler.get_last_lr()[0]:.2e}")

        # 保存最新
        ckpt = {
            "epoch":        epoch,
            "state_dict":   model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "val_acc":      val_acc,
            "best_val_acc": best_val_acc,
            "classes":      LABEL_NAMES,
            "num_classes":  args.num_classes,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "shoe_last.pth"))

        # 保存最佳
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            ckpt["best_val_acc"] = best_val_acc
            torch.save(ckpt, os.path.join(args.output_dir, "shoe_best.pth"))
            print(f"  ★ 新最佳: {best_val_acc:.2f}%  → 已保存 shoe_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⚠ 早停 (连续 {patience} 轮无改善)")
                break

    # ── 导出模型信息 ──────────────────────────────────────
    info = {
        "classes":      {i: n for i, n in enumerate(LABEL_NAMES)},
        "num_classes":  args.num_classes,
        "architecture": "resnet50",
        "input_size":   [224, 224],
        "best_val_acc": best_val_acc,
        "preprocessing": {
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225],
        },
        "label_remap_note": "训练类别 0=sandals 1=low_shoes 2=boots（原始ID 0/1/3）",
    }
    info_path = os.path.join(args.output_dir, "shoe_model_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print(f"权重输出目录: {args.output_dir}/")
    print(f"  shoe_best.pth  ← 最佳模型（下载后使用）")
    print(f"  shoe_last.pth  ← 最新 checkpoint")
    print(f"  shoe_model_info.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
