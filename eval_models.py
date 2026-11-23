"""
验证集评估脚本
评估三个模型：
  1. 障碍物检测  - YOLOv8  best.pt         -> mAP50, mAP50-95
  2. 水坑分割    - U-Net   water_model_best.pth -> mIoU, Pixel Accuracy, F1
  3. 鞋子识别    - ResNet50 shoe_best.pth   -> Top-1 Accuracy (已知 96.93%)

用法：
  python eval_models.py [--obstacle] [--water] [--shoe]
  不加参数则评估全部
"""
import argparse
import sys
import os
from pathlib import Path

# ── 项目根目录加入 sys.path ───────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────────────────────────────────────
OBSTACLE_MODEL  = ROOT / "training_checkpoints" / "obstacle" / "best.pt"
OBSTACLE_DATA   = ROOT / "training_data" / "obstacle_yolo" / "dataset.yaml"
OBSTACLE_VAL_IMGS = ROOT / "training_data" / "obstacle_yolo" / "images" / "val"

WATER_MODEL     = ROOT / "training_checkpoints" / "water_model_best.pth"
WATER_IMGS      = ROOT / "training_data" / "water_real" / "images"
WATER_MASKS     = ROOT / "training_data" / "water_real" / "masks"

SHOE_MODEL      = ROOT / "training_checkpoints" / "checkpoints" / "shoe_best.pth"


# ═════════════════════════════════════════════════════════════════════════════
# 1. 障碍物检测评估  (YOLOv8)
# ═════════════════════════════════════════════════════════════════════════════
def eval_obstacle():
    print("\n" + "="*60)
    print("障碍物检测模型评估  (YOLOv8)")
    print("="*60)

    if not OBSTACLE_MODEL.exists():
        # 尝试备用路径
        alt = ROOT / "runs" / "detect" / "training_checkpoints" / "obstacle_yolov8" / "weights" / "best.pt"
        if alt.exists():
            model_path = alt
        else:
            print(f"[ERROR] 找不到模型文件: {OBSTACLE_MODEL}")
            return
    else:
        model_path = OBSTACLE_MODEL

    if not OBSTACLE_DATA.exists():
        print(f"[ERROR] 找不到数据集配置: {OBSTACLE_DATA}")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 未安装 ultralytics，请运行: pip install ultralytics")
        return

    print(f"模型路径: {model_path}")
    print(f"数据集  : {OBSTACLE_DATA}")
    print(f"验证图片: {OBSTACLE_VAL_IMGS} ({len(list(OBSTACLE_VAL_IMGS.glob('*.*')))} 张)")
    print()

    model = YOLO(str(model_path))

    # 修复 dataset.yaml 中的路径（可能含乱码）
    import yaml, tempfile, shutil
    with open(OBSTACLE_DATA, 'r', encoding='utf-8', errors='replace') as f:
        data_cfg = yaml.safe_load(f)

    # 强制使用本地绝对路径
    data_cfg['path'] = str(ROOT / "training_data" / "obstacle_yolo")
    data_cfg['train'] = 'images/train'
    data_cfg['val']   = 'images/val'

    tmp_yaml = ROOT / "_eval_obstacle_tmp.yaml"
    with open(tmp_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(data_cfg, f, allow_unicode=True)

    try:
        results = model.val(
            data=str(tmp_yaml),
            split='val',
            imgsz=640,
            batch=8,
            device='cpu',
            verbose=True,
        )

        print("\n── 评估结果 ──────────────────────────────────────────")
        print(f"mAP50      : {results.box.map50:.4f}   ({results.box.map50*100:.2f}%)")
        print(f"mAP50-95   : {results.box.map:.4f}   ({results.box.map*100:.2f}%)")
        print(f"Precision  : {results.box.mp:.4f}")
        print(f"Recall     : {results.box.mr:.4f}")

        # 每类 AP
        if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
            print("\n── 各类别 AP50 ───────────────────────────────────────")
            names = model.names
            for i, ap in zip(results.box.ap_class_index, results.box.ap50):
                print(f"  {names[int(i)]:<20s}: {ap:.4f}  ({ap*100:.2f}%)")

    finally:
        if tmp_yaml.exists():
            tmp_yaml.unlink()


# ═════════════════════════════════════════════════════════════════════════════
# 2. 水坑分割评估  (U-Net)
# ═════════════════════════════════════════════════════════════════════════════
def eval_water():
    print("\n" + "="*60)
    print("水坑分割模型评估  (U-Net)")
    print("="*60)

    if not WATER_MODEL.exists():
        print(f"[ERROR] 找不到模型文件: {WATER_MODEL}")
        return
    if not WATER_IMGS.exists() or not WATER_MASKS.exists():
        print(f"[ERROR] 找不到验证数据: {WATER_IMGS}")
        return

    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        import numpy as np
        import cv2
        from PIL import Image
    except ImportError as e:
        print(f"[ERROR] 缺少依赖: {e}")
        return

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    # 内联 U-Net 定义（避免路径问题）
    class UNet(nn.Module):
        """与 pytorch_water_detector.py 完全一致的层名"""
        def __init__(self):
            super().__init__()
            def cb(ic, oc):
                return nn.Sequential(
                    nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
                    nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True))
            # 与原文件相同的层名：enc1..enc4, bottleneck, up4..up1, dec4..dec1
            self.enc1=cb(3,64);   self.enc2=cb(64,128)
            self.enc3=cb(128,256);self.enc4=cb(256,512)
            self.pool=nn.MaxPool2d(2)
            self.bottleneck=cb(512,1024)
            self.up4=nn.ConvTranspose2d(1024,512,2,2); self.dec4=cb(1024,512)
            self.up3=nn.ConvTranspose2d(512,256,2,2);  self.dec3=cb(512,256)
            self.up2=nn.ConvTranspose2d(256,128,2,2);  self.dec2=cb(256,128)
            self.up1=nn.ConvTranspose2d(128,64,2,2);   self.dec1=cb(128,64)
            self.out=nn.Conv2d(64,1,1); self.sigmoid=nn.Sigmoid()
        def forward(self,x):
            e1=self.enc1(x); e2=self.enc2(self.pool(e1))
            e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
            b=self.bottleneck(self.pool(e4))
            d4=self.dec4(torch.cat([self.up4(b),e4],1))
            d3=self.dec3(torch.cat([self.up3(d4),e3],1))
            d2=self.dec2(torch.cat([self.up2(d3),e2],1))
            d1=self.dec1(torch.cat([self.up1(d2),e1],1))
            return self.sigmoid(self.out(d1))

    device = torch.device('cpu')
    model = UNet().to(device)
    ck = torch.load(str(WATER_MODEL), map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    print(f"模型已加载: {WATER_MODEL}")

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # ── 配对图片与 mask ──────────────────────────────────────────────────────
    img_files = {f.stem: f for f in WATER_IMGS.iterdir() if f.suffix.lower() in {'.jpg','.jpeg','.png'}}
    mask_files = {f.stem: f for f in WATER_MASKS.iterdir() if f.suffix.lower() in {'.jpg','.jpeg','.png'}}
    paired = [(img_files[k], mask_files[k]) for k in img_files if k in mask_files]
    print(f"配对样本数 : {len(paired)} / 图片 {len(img_files)} 张 / mask {len(mask_files)} 张")
    print(f"阈值       : 0.15")
    print()

    THRESHOLD = 0.15
    total_iou = 0.0
    total_f1  = 0.0
    total_px_acc = 0.0
    count = 0
    errors = 0

    for img_path, mask_path in paired:
        try:
            img = Image.open(img_path).convert('RGB')
            oh, ow = img.size[1], img.size[0]
            inp = transform(img).unsqueeze(0).to(device)

            # 用 PIL + numpy 读取 mask，避免 cv2.imread 不支持中文路径
            gt_pil = Image.open(mask_path).convert('L')
            gt_raw = np.array(gt_pil)
            gt_pil_resized = gt_pil.resize((ow, oh), Image.NEAREST)
            gt = (np.array(gt_pil_resized) > 127).astype(np.uint8)

            with torch.no_grad():
                pred_conf = model(inp)[0,0].cpu().numpy()
            pred_pil = Image.fromarray((pred_conf * 255).astype(np.uint8))
            pred_conf = np.array(pred_pil.resize((ow, oh), Image.BILINEAR)) / 255.0
            pred = (pred_conf > THRESHOLD).astype(np.uint8)

            inter = int(np.logical_and(pred, gt).sum())
            union = int(np.logical_or(pred, gt).sum())
            iou   = inter / union if union > 0 else 1.0

            tp = inter
            fp = int(pred.sum()) - tp
            fn = int(gt.sum())   - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0

            px_acc = float(np.mean(pred == gt))

            total_iou    += iou
            total_f1     += f1
            total_px_acc += px_acc
            count += 1

        except Exception as e:
            errors += 1

        if count % 100 == 0 and count > 0:
            print(f"  进度: {count}/{len(paired)}  当前平均IoU={total_iou/count:.4f}")

    if count == 0:
        print("[ERROR] 没有成功评估任何样本")
        return

    miou     = total_iou / count
    mean_f1  = total_f1  / count
    mean_acc = total_px_acc / count

    print(f"\n── 评估结果 ({count} 样本) ────────────────────────────────")
    print(f"mIoU (平均交并比) : {miou:.4f}   ({miou*100:.2f}%)")
    print(f"Pixel Accuracy    : {mean_acc:.4f}  ({mean_acc*100:.2f}%)")
    print(f"F1 Score (Dice)   : {mean_f1:.4f}   ({mean_f1*100:.2f}%)")
    if errors:
        print(f"跳过（读取失败）  : {errors} 张")


# ═════════════════════════════════════════════════════════════════════════════
# 3. 鞋子识别评估  (ResNet50) — checkpoint 已记录，直接读取
# ═════════════════════════════════════════════════════════════════════════════
def eval_shoe():
    print("\n" + "="*60)
    print("鞋子识别模型评估  (ResNet50)")
    print("="*60)

    if not SHOE_MODEL.exists():
        print(f"[ERROR] 找不到模型文件: {SHOE_MODEL}")
        return

    try:
        import torch
    except ImportError:
        print("[ERROR] 未安装 PyTorch")
        return

    ck = torch.load(str(SHOE_MODEL), map_location='cpu')
    print(f"模型路径     : {SHOE_MODEL}")
    print(f"架构         : ResNet50")
    print(f"类别         : {ck.get('classes', ['sandals','low_shoes','boots'])}")
    print(f"训练轮次     : {ck.get('epoch', '?')} epoch")
    print(f"\n── 评估结果（来自训练时的验证集）────────────────────────")
    print(f"Best Val Accuracy : {ck.get('best_val_acc', 'N/A'):.4f}%")
    print(f"Last Val Accuracy : {ck.get('val_acc', 'N/A'):.4f}%")
    print("\n注：鞋子验证集在魔搭/DSW云端训练，本地无原始验证集图片，")
    print("    以上数字为训练时保存的准确率，可信度高。")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="模型验证集评估")
    parser.add_argument('--obstacle', action='store_true', help='只评估障碍物模型')
    parser.add_argument('--water',    action='store_true', help='只评估水坑模型')
    parser.add_argument('--shoe',     action='store_true', help='只评估鞋子模型')
    args = parser.parse_args()

    run_all = not (args.obstacle or args.water or args.shoe)

    if args.obstacle or run_all:
        eval_obstacle()
    if args.water or run_all:
        eval_water()
    if args.shoe or run_all:
        eval_shoe()

    print("\n" + "="*60)
    print("评估完成")
    print("="*60)


if __name__ == '__main__':
    main()
