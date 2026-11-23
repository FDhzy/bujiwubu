"""
YOLOv8 水坑分割训练脚本
任务类型: segment（实例分割）

数据来源:
    training_data/water_real/images/  ← 原始图片
    training_data/water_real/masks/   ← 二值掩码（白色=水坑）

自动完成:
    1. 将 mask 转换为 YOLO segment 格式（.txt 多边形标注）
    2. 按 val_ratio 划分 train/val
    3. 生成 dataset.yaml
    4. 调用 YOLOv8 segment 训练

用法:
    python train_water_yolov8.py                  # 默认参数
    python train_water_yolov8.py --skip-build     # 跳过数据集构建
    python train_water_yolov8.py --epochs 100 --model-size s
"""
import os
import sys
import shutil
import random
import argparse
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
SRC_IMAGES = ROOT / "training_data" / "water_real" / "images"
SRC_MASKS  = ROOT / "training_data" / "water_real" / "masks"
YOLO_DATA  = ROOT / "training_data" / "water_yolov8"
OUTPUT_DIR = ROOT / "training_checkpoints" / "water_yolov8"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── 掩码 → YOLO segment 标注 ─────────────────────────────────────────────────

def mask_to_yolo_segments(mask_path: Path, img_w: int, img_h: int,
                           min_area: int = 200) -> list[str]:
    """
    将二值掩码转换为 YOLO segment 格式的标注行。
    每个连通域生成一行: 0 x1 y1 x2 y2 ... xn yn （归一化坐标）

    Args:
        mask_path: 掩码文件路径
        img_w, img_h: 对应图片的宽高
        min_area: 忽略面积小于此值的噪点区域（像素数）
    Returns:
        list of YOLO segment annotation lines
    """
    mask_bgr = cv2.imread(str(mask_path))
    if mask_bgr is None:
        # 尝试 PIL 读取（处理中文路径）
        from PIL import Image
        pil = Image.open(mask_path).convert("L")
        gray = np.array(pil)
    else:
        gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)

    # 二值化：白色为水坑
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 可选：resize mask 到图片尺寸（防止不一致）
    mh, mw = binary.shape[:2]
    if mh != img_h or mw != img_w:
        binary = cv2.resize(binary, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 简化轮廓减少点数（epsilon = 轮廓周长的 0.5%）
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            continue

        # 归一化坐标
        coords = []
        for pt in approx.reshape(-1, 2):
            x_norm = float(np.clip(pt[0] / img_w, 0.0, 1.0))
            y_norm = float(np.clip(pt[1] / img_h, 0.0, 1.0))
            coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])

        # YOLO segment 格式: class_id x1 y1 x2 y2 ...
        lines.append("0 " + " ".join(coords))

    return lines


# ── 配对图片和掩码 ────────────────────────────────────────────────────────────

def find_pairs() -> list[tuple[Path, Path]]:
    """返回 (image_path, mask_path) 配对列表。"""
    pairs = []
    mask_stems = {p.stem: p for p in SRC_MASKS.iterdir()
                  if p.suffix.lower() in IMG_EXTS}

    for img_path in SRC_IMAGES.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        mask_path = mask_stems.get(img_path.stem)
        if mask_path is None:
            continue
        pairs.append((img_path, mask_path))

    return pairs


# ── 构建数据集 ────────────────────────────────────────────────────────────────

def build_dataset(val_ratio: float = 0.15, seed: int = 42):
    """将 water_real 数据转换为 YOLOv8 segment 格式。"""
    random.seed(seed)

    pairs = find_pairs()
    if not pairs:
        print(f"[ERROR] 未找到图片-掩码配对")
        print(f"  图片目录: {SRC_IMAGES}")
        print(f"  掩码目录: {SRC_MASKS}")
        sys.exit(1)

    print(f"找到 {len(pairs)} 对图片/掩码")

    # 清理并重建目录
    if YOLO_DATA.exists():
        shutil.rmtree(YOLO_DATA)
    for split in ("train", "val"):
        (YOLO_DATA / split / "images").mkdir(parents=True, exist_ok=True)
        (YOLO_DATA / split / "labels").mkdir(parents=True, exist_ok=True)

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]

    skipped = 0

    def process_split(split_pairs, split_name):
        nonlocal skipped
        img_dir = YOLO_DATA / split_name / "images"
        lbl_dir = YOLO_DATA / split_name / "labels"
        for img_path, mask_path in split_pairs:
            # 读图获取尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                from PIL import Image as PILImage
                try:
                    pil = PILImage.open(img_path)
                    img_w, img_h = pil.size
                except Exception:
                    skipped += 1
                    continue
            else:
                img_h, img_w = img.shape[:2]

            # 生成标注
            lines = mask_to_yolo_segments(mask_path, img_w, img_h)
            if not lines:
                # 无有效水坑区域，跳过（避免引入空标注影响训练）
                skipped += 1
                continue

            # 复制图片
            dst_img = img_dir / img_path.name
            shutil.copy2(img_path, dst_img)

            # 写标注文件
            lbl_file = lbl_dir / (img_path.stem + ".txt")
            lbl_file.write_text("\n".join(lines), encoding="utf-8")

    process_split(train_pairs, "train")
    process_split(val_pairs,   "val")

    train_cnt = len(list((YOLO_DATA / "train" / "images").iterdir()))
    val_cnt   = len(list((YOLO_DATA / "val"   / "images").iterdir()))
    print(f"数据集构建完成: train={train_cnt}  val={val_cnt}  跳过={skipped}")

    # 生成 dataset.yaml
    yaml_path = YOLO_DATA / "dataset.yaml"
    yaml_content = f"""# YOLOv8 water puddle segmentation dataset
path: {YOLO_DATA.as_posix()}
train: train/images
val:   val/images

nc: 1
names:
  0: water_puddle
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"dataset.yaml 已生成: {yaml_path}")
    return yaml_path


# ── 训练 ──────────────────────────────────────────────────────────────────────

def train(yaml_path: Path, epochs: int = 100, imgsz: int = 640,
          batch: int = 8, model_size: str = "n", device: str = ""):
    """调用 YOLOv8 segment 训练。"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 未安装 ultralytics，请执行: pip install ultralytics")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_model = f"yolov8{model_size}-seg.pt"
    print(f"\n{'='*55}")
    print(f"YOLOv8 水坑分割训练")
    print(f"  基础模型: {base_model}")
    print(f"  数据集:   {yaml_path}")
    print(f"  epochs:   {epochs}  imgsz: {imgsz}  batch: {batch}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'='*55}\n")

    model = YOLO(base_model)
    model.train(
        data=str(yaml_path),
        task="segment",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device if device else None,
        project=str(OUTPUT_DIR.parent),
        name="water_yolov8",
        exist_ok=True,
        patience=20,
        # 数据增强（水坑场景适用）
        augment=True,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        flipud=0.3,    # 上下翻转（模拟倒影）
        fliplr=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        mosaic=0.5,
    )

    best_pt = OUTPUT_DIR / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n训练完成！最佳模型: {best_pt}")
        print(f"config 中 yolo_water_model_path 已指向此路径，重启应用即可使用。")
    else:
        print(f"\n训练完成，未找到 best.pt，请检查: {OUTPUT_DIR}")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 水坑分割训练")
    parser.add_argument("--skip-build", action="store_true",
                        help="跳过数据集构建，直接训练（需已有 training_data/water_yolov8/）")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--imgsz",      type=int,   default=640,
                        help="训练图像尺寸（推荐 640）")
    parser.add_argument("--batch",      type=int,   default=8)
    parser.add_argument("--model-size", type=str,   default="n",
                        choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--device",     type=str,   default="",
                        help="训练设备，如 0 (GPU0), cpu，留空自动选择")
    parser.add_argument("--val-ratio",  type=float, default=0.15)
    args = parser.parse_args()

    os.chdir(ROOT)

    yaml_path = YOLO_DATA / "dataset.yaml"

    if not args.skip_build:
        print("── 步骤 1: 构建 YOLOv8 分割数据集 ──")
        yaml_path = build_dataset(val_ratio=args.val_ratio)
    else:
        print("── 跳过数据集构建 ──")
        if not yaml_path.exists():
            print(f"[ERROR] dataset.yaml 不存在: {yaml_path}")
            sys.exit(1)

    print("\n── 步骤 2: 开始训练 ──")
    train(
        yaml_path=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_size=args.model_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
