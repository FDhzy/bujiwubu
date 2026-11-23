"""
YOLOv8 鞋型分类训练脚本
任务类型: classify（图像分类）
训练数据结构要求（ImageFolder 格式）:
    training_data/shoe_yolov8/
        train/
            sandals/       ← 凉鞋/拖鞋
            low_shoes/     ← 低帮鞋/运动鞋
            ankle_shoes/   ← 踝靴
            boots/         ← 中筒靴
            high_boots/    ← 雨靴/高筒靴
        val/
            sandals/
            low_shoes/
            ...

用法:
    python train_shoe_yolov8.py               # 从现有 shoe_real 数据集重组并训练
    python train_shoe_yolov8.py --skip-build  # 跳过数据重组，直接训练
    python train_shoe_yolov8.py --epochs 50   # 自定义 epoch 数
"""
import os
import sys
import shutil
import random
import argparse
from pathlib import Path

# ── 路径常量 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
SRC_DIR     = ROOT / "training_data" / "shoe_real" / "images"
SRC_LABELS  = ROOT / "training_data" / "shoe_real" / "labels"
SRC_MAP     = ROOT / "training_data" / "shoe_real" / "category_mapping.json"
YOLO_DATA   = ROOT / "training_data" / "shoe_yolov8"
OUTPUT_DIR  = ROOT / "training_checkpoints" / "shoe_yolov8"

# 5 个目标类别（文件夹名需与 _NAME_TO_TYPE 映射表一致）
TARGET_CLASSES = ["sandals", "low_shoes", "ankle_shoes", "boots", "high_boots"]

# 原始数据集类别 → 目标类别
LABEL_REMAP = {
    "Shoes":    "low_shoes",
    "Sandals":  "sandals",
    "Slippers": "sandals",
    "Boots":    "boots",
}

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def read_yolo_label(label_path: Path) -> str:
    """读取 YOLO 格式 label 文件，返回第一行的类别 ID（字符串）。"""
    try:
        text = label_path.read_text().strip()
        if text:
            return text.split()[0]
    except Exception:
        pass
    return ""


def build_dataset(val_ratio: float = 0.15, seed: int = 42):
    """
    将 shoe_real/images + shoe_real/labels 重组为 YOLOv8 classify 文件夹结构。
    Label 文件里的类别 ID 对应 category_mapping.json 的 source_categories 顺序。
    """
    import json
    random.seed(seed)

    # 读取映射
    with open(SRC_MAP, encoding="utf-8") as f:
        mapping_data = json.load(f)
    src_classes = mapping_data["source_categories"]   # 下标即类别 ID
    # idx -> target_class_name
    idx_to_target = {
        str(i): LABEL_REMAP.get(name, None)
        for i, name in enumerate(src_classes)
    }

    # 收集所有图片及对应目标类别
    samples = []  # list of (img_path, target_class)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for img_path in SRC_DIR.iterdir():
        if img_path.suffix.lower() not in img_exts:
            continue
        label_path = SRC_LABELS / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        cls_id = read_yolo_label(label_path)
        target = idx_to_target.get(cls_id)
        if target is None:
            continue
        samples.append((img_path, target))

    if not samples:
        print("[ERROR] 未找到任何带标注的图片，请检查:")
        print(f"  图片目录: {SRC_DIR}")
        print(f"  标注目录: {SRC_LABELS}")
        sys.exit(1)

    print(f"共找到 {len(samples)} 张有效图片")

    # 按类别统计
    from collections import Counter
    cnt = Counter(t for _, t in samples)
    for cls, n in sorted(cnt.items()):
        print(f"  {cls}: {n} 张")

    # 清理并重建目录
    if YOLO_DATA.exists():
        shutil.rmtree(YOLO_DATA)
    for split in ("train", "val"):
        for cls in TARGET_CLASSES:
            (YOLO_DATA / split / cls).mkdir(parents=True, exist_ok=True)

    # 按类别分组后 shuffle + split
    from collections import defaultdict
    by_class = defaultdict(list)
    for img_path, target in samples:
        by_class[target].append(img_path)

    total_train, total_val = 0, 0
    for cls, paths in by_class.items():
        random.shuffle(paths)
        n_val = max(1, int(len(paths) * val_ratio))
        val_paths   = paths[:n_val]
        train_paths = paths[n_val:]
        for p in train_paths:
            shutil.copy2(p, YOLO_DATA / "train" / cls / p.name)
        for p in val_paths:
            shutil.copy2(p, YOLO_DATA / "val" / cls / p.name)
        total_train += len(train_paths)
        total_val   += len(val_paths)
        print(f"  {cls}: train={len(train_paths)}, val={len(val_paths)}")

    print(f"\n数据集构建完成: train={total_train}, val={total_val}")
    print(f"保存至: {YOLO_DATA}")


def train(epochs: int = 50, imgsz: int = 224, batch: int = 16,
          model_size: str = "n", device: str = ""):
    """
    用 YOLOv8 classify 任务训练鞋型分类模型。

    Args:
        epochs:     训练轮数
        imgsz:      输入图像尺寸
        batch:      批大小
        model_size: 模型规模 n/s/m/l/x
        device:     训练设备，空字符串=自动选择
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 未安装 ultralytics，请执行: pip install ultralytics")
        sys.exit(1)

    if not YOLO_DATA.exists():
        print(f"[ERROR] 数据集目录不存在: {YOLO_DATA}")
        print("请先运行 python train_shoe_yolov8.py 构建数据集")
        sys.exit(1)

    # 检查 val 集
    val_dir = YOLO_DATA / "val"
    if not val_dir.exists() or not any(val_dir.iterdir()):
        print(f"[ERROR] val 目录为空: {val_dir}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_model = f"yolov8{model_size}-cls.pt"
    print(f"\n{'='*50}")
    print(f"YOLOv8 鞋型分类训练")
    print(f"  基础模型: {base_model}")
    print(f"  数据集:   {YOLO_DATA}")
    print(f"  epochs:   {epochs}")
    print(f"  imgsz:    {imgsz}")
    print(f"  batch:    {batch}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'='*50}\n")

    model = YOLO(base_model)
    results = model.train(
        data=str(YOLO_DATA),
        task="classify",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device if device else None,
        project=str(OUTPUT_DIR.parent),
        name="shoe_yolov8",
        exist_ok=True,
        patience=15,          # early stopping
        augment=True,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.5,
        degrees=10.0,
    )

    # 找到 best.pt
    best_pt = OUTPUT_DIR / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n训练完成！最佳模型: {best_pt}")
        print(f"将模型路径写入 config 中的 yolo_shoe_model_path = '{best_pt}'")
    else:
        print("\n训练完成，但未找到 best.pt，请检查输出目录:", OUTPUT_DIR)

    return results


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 鞋型分类训练")
    parser.add_argument("--skip-build", action="store_true",
                        help="跳过数据集重组，直接训练")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--imgsz",      type=int,   default=224)
    parser.add_argument("--batch",      type=int,   default=16)
    parser.add_argument("--model-size", type=str,   default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 模型规模: n(最小/最快) ~ x(最大/最准)")
    parser.add_argument("--device",     type=str,   default="",
                        help="训练设备，如 0 (GPU0), cpu, 留空自动选择")
    parser.add_argument("--val-ratio",  type=float, default=0.15,
                        help="验证集比例，默认 0.15")
    args = parser.parse_args()

    # 切换到项目根目录，避免相对路径问题
    os.chdir(ROOT)

    if not args.skip_build:
        print("── 步骤 1: 构建 YOLOv8 分类数据集 ──")
        build_dataset(val_ratio=args.val_ratio)
    else:
        print("── 跳过数据集构建 ──")

    print("\n── 步骤 2: 开始训练 ──")
    train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_size=args.model_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
