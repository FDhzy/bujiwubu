"""
障碍物 YOLOv8 合并训练脚本
=============================
功能：
  1. 读取原有 obstacle_yolo/ 数据集（7类，已划分 train/val）
  2. 读取新增数据（车辆/ 行人/ 台阶/ 自行车/），修正 class id 映射
  3. 对新数据进行过采样（复制 repeat_times 份），强化训练
  4. 合并为新数据集，重新划分 train/val
  5. 调用 YOLOv8 detect 训练

class id 映射（新数据标错，脚本自动纠正）：
  车辆/  : txt 中 id=3 → 重映射为 2 (car)
  行人/  : txt 中 id=4 → 重映射为 0 (person)
  台阶/  : txt 中 id=0 → 重映射为 4 (stairs)，多边形格式自动转 bbox
  自行车/: txt 中 id=0 → 重映射为 1 (bicycle)

类别体系（沿用原 obstacle_yolo 7类）：
  0: person  1: bicycle  2: car  3: motorcycle
  4: stairs  5: wire     6: bench

用法：
  python merge_and_train_obstacle.py
  python merge_and_train_obstacle.py --repeat 8 --epochs 150 --model-size s
  python merge_and_train_obstacle.py --skip-build  # 跳过合并，直接训练
"""
import os
import sys
import shutil
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).parent

# ── 原有数据集（已有 train/val 划分）────────────────────────────────────────
ORIG_YOLO_DIR = ROOT / "training_data" / "obstacle_yolo"

# ── 今日新增数据（图片 + txt 同目录，class id 需重映射）───────────────────────
# 格式：(目录路径, {旧class_id: 新class_id, ...})
# 台阶数据：图片在 台阶/ 子目录，标注在 mask/ 子目录（多边形格式，自动转 bbox）
# 自行车数据：图片在 bikes/ 子目录，标注在 bicycle_annotations/ 子目录
NEW_DATA_SOURCES = [
    (ROOT / "training_data" / "车辆",  {3: 2}),      # 车辆 → car(2)
    (ROOT / "training_data" / "行人",  {4: 0}),      # 行人 → person(0)
    # 台阶：图片目录 / 标注目录 分开，特殊格式，由 add_stairs_data() 处理
    # 自行车：图片目录 / 标注目录 分开，由 add_bicycle_data() 处理
]

# ── 台阶数据（图片与标注在不同子目录）────────────────────────────────────────
STAIRS_IMG_DIR = ROOT / "training_data" / "电线杆+台阶" / "台阶"
STAIRS_LBL_DIR = ROOT / "training_data" / "电线杆+台阶" / "mask"

# ── 自行车数据（图片与标注在不同子目录）──────────────────────────────────────
BICYCLE_IMG_DIR = ROOT / "training_data" / "bicycle" / "bikes"
BICYCLE_LBL_DIR = ROOT / "training_data" / "bicycle" / "bicycle_annotations"

# ── 合并后的输出数据集目录 ───────────────────────────────────────────────────
MERGED_DIR  = ROOT / "training_data" / "obstacle_merged"
OUTPUT_DIR  = ROOT / "training_checkpoints" / "obstacle_merged"

# ── 类别定义（与原 obstacle_yolo 保持一致）───────────────────────────────────
CLASS_NAMES = [
    "person",     # 0
    "bicycle",    # 1
    "car",        # 2
    "motorcycle", # 3
    "stairs",     # 4
    "wire",       # 5
    "bench",      # 6
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def remap_label_file(src_txt: Path, dst_txt: Path, id_map: dict):
    """
    读取 YOLO bbox 标注文件，将 class id 按 id_map 重映射后写出。
    id_map 中未出现的 id 保持不变。
    空文件（无标注）原样复制。
    """
    lines = src_txt.read_text(encoding="utf-8").strip().splitlines()
    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_id = int(parts[0])
        new_id = id_map.get(old_id, old_id)
        new_lines.append(f"{new_id} {' '.join(parts[1:])}")
    dst_txt.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def polygon_to_bbox_line(line: str, new_class_id: int) -> str:
    """
    将 YOLO 分割多边形标注行转换为检测 bbox 标注行。
    输入格式: "old_id x1 y1 x2 y2 x3 y3 ..."（多边形顶点，归一化坐标）
    输出格式: "new_class_id cx cy w h"（外接矩形，归一化坐标）
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return ""
    coords = list(map(float, parts[1:]))
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min
    # 限制在 [0,1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.001, min(1.0, w))
    h  = max(0.001, min(1.0, h))
    return f"{new_class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_polygon_label(src_txt: Path, dst_txt: Path, new_class_id: int):
    """读取多边形标注文件，转换为 bbox 格式后写出（跳过坐标不足的行）。"""
    lines = src_txt.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # 检测格式（5列）直接重映射 class id
        if len(parts) == 5:
            new_lines.append(f"{new_class_id} {' '.join(parts[1:])}")
        elif len(parts) >= 9:
            # 多边形格式：至少4个点（8个坐标）
            converted = polygon_to_bbox_line(line, new_class_id)
            if converted:
                new_lines.append(converted)
    if new_lines:
        dst_txt.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    else:
        dst_txt.write_text("", encoding="utf-8")


def find_image_label_pairs(directory: Path):
    """返回目录下所有 (image_path, label_path) 配对（label 必须存在）。"""
    pairs = []
    for img_path in directory.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            pairs.append((img_path, txt_path))
    return pairs


def find_cross_dir_pairs(img_dir: Path, lbl_dir: Path):
    """
    图片和标注在不同目录时的配对查找。
    图片在 img_dir，对应的 .txt 在 lbl_dir（stem 相同）。
    """
    pairs = []
    if not img_dir.exists() or not lbl_dir.exists():
        return pairs
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        # 尝试各种可能的图片后缀对应的 stem
        stem = img_path.stem
        # 处理文件名含多个点的情况（如 "abc.jpg_wh860.jpg" → stem="abc.jpg_wh860"）
        txt_path = lbl_dir / (stem + ".txt")
        if txt_path.exists():
            pairs.append((img_path, txt_path))
    return pairs


def copy_orig_dataset(merged_dir: Path):
    """
    将原有 obstacle_yolo 数据集（已有 train/val 结构）原样复制到合并目录。
    保留原始 class id，不做任何修改。
    """
    print(f"── 复制原始数据集: {ORIG_YOLO_DIR}")
    total = 0
    for split in ("train", "val"):
        src_img_dir = ORIG_YOLO_DIR / "images" / split
        src_lbl_dir = ORIG_YOLO_DIR / "labels" / split
        dst_img_dir = merged_dir / "images" / split
        dst_lbl_dir = merged_dir / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            print(f"  [WARN] 目录不存在: {src_img_dir}")
            continue

        for img_path in src_img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl_dir / (img_path.stem + ".txt"))
            total += 1

    print(f"   原始数据集复制完成，共 {total} 张图片")
    return total


def add_new_data(merged_dir: Path, repeat_times: int, val_ratio: float, seed: int = 42):
    """
    将 NEW_DATA_SOURCES 中的新数据重映射 class id 后过采样加入合并目录。
    同时处理台阶（多边形→bbox）和自行车（跨目录配对）。
    新数据全部先汇总，再按 val_ratio 划分 train/val。
    """
    import tempfile
    random.seed(seed)

    # 汇总所有新数据对：(img_path, txt_path_or_none, id_map_or_none, is_polygon, is_cross_dir_stairs)
    # 统一格式: (img_path, txt_path, id_map, is_polygon)
    all_new_pairs = []

    # 1. 常规数据源（图片和标注同目录）
    for data_dir, id_map in NEW_DATA_SOURCES:
        if not data_dir.exists():
            print(f"  [WARN] 新数据目录不存在，跳过: {data_dir}")
            continue
        pairs = find_image_label_pairs(data_dir)
        for img, txt in pairs:
            all_new_pairs.append((img, txt, id_map, False))
        print(f"   {data_dir.name}: {len(pairs)} 张")

    # 2. 台阶数据（多边形标注，图片和标注在不同子目录）
    if STAIRS_IMG_DIR.exists() and STAIRS_LBL_DIR.exists():
        stairs_pairs = find_cross_dir_pairs(STAIRS_IMG_DIR, STAIRS_LBL_DIR)
        for img, txt in stairs_pairs:
            all_new_pairs.append((img, txt, {0: 4}, True))  # 多边形，class 0→stairs(4)
        print(f"   台阶(电线杆+台阶/台阶): {len(stairs_pairs)} 张")
    else:
        print(f"  [WARN] 台阶目录不存在，跳过: {STAIRS_IMG_DIR}")

    # 3. 自行车数据（图片和标注在不同子目录，bbox 格式）
    if BICYCLE_IMG_DIR.exists() and BICYCLE_LBL_DIR.exists():
        bicycle_pairs = find_cross_dir_pairs(BICYCLE_IMG_DIR, BICYCLE_LBL_DIR)
        for img, txt in bicycle_pairs:
            all_new_pairs.append((img, txt, {0: 1}, False))  # class 0→bicycle(1)
        print(f"   自行车(bicycle/bikes): {len(bicycle_pairs)} 张")
    else:
        print(f"  [WARN] 自行车目录不存在，跳过: {BICYCLE_IMG_DIR}")

    if not all_new_pairs:
        print("[WARN] 未找到任何新数据，将仅训练原始数据集")
        return 0

    print(f"\n新数据汇总: {len(all_new_pairs)} 张，过采样 × {repeat_times}")

    # 打乱后划分 train/val
    random.shuffle(all_new_pairs)
    n_val = max(1, int(len(all_new_pairs) * val_ratio))
    val_pairs   = all_new_pairs[:n_val]
    train_pairs = all_new_pairs[n_val:]

    def write_split(pairs, split_name, times):
        img_dir = merged_dir / "images" / split_name
        lbl_dir = merged_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        copy_times = times if split_name == "train" else 1  # val 不复制
        for t in range(copy_times):
            suffix = f"_aug{t}" if t > 0 else ""
            for img_path, txt_path, id_map, is_polygon in pairs:
                dst_stem = img_path.stem + suffix
                # 统一使用 .jpg 后缀避免 webp/avif 等格式问题（复制原文件）
                dst_img  = img_dir / (dst_stem + img_path.suffix)
                dst_lbl  = lbl_dir / (dst_stem + ".txt")
                shutil.copy2(img_path, dst_img)
                if is_polygon:
                    # 台阶：多边形格式转 bbox
                    new_class_id = list(id_map.values())[0]
                    convert_polygon_label(txt_path, dst_lbl, new_class_id)
                else:
                    remap_label_file(txt_path, dst_lbl, id_map)
                count += 1
        return count

    train_cnt = write_split(train_pairs, "train", repeat_times)
    val_cnt   = write_split(val_pairs,   "val",   1)

    print(f"   新数据写入: train={train_cnt}  val={val_cnt}")
    return train_cnt + val_cnt


def count_split(merged_dir: Path, split: str) -> int:
    d = merged_dir / "images" / split
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)


def build_merged_dataset(repeat_times: int, val_ratio: float):
    """完整构建合并数据集。"""
    # 清理旧合并目录
    if MERGED_DIR.exists():
        print(f"清理旧合并数据集: {MERGED_DIR}")
        shutil.rmtree(MERGED_DIR)
    MERGED_DIR.mkdir(parents=True)

    # 复制原始数据
    copy_orig_dataset(MERGED_DIR)

    # 加入新数据（过采样）
    add_new_data(MERGED_DIR, repeat_times=repeat_times, val_ratio=val_ratio)

    # 统计结果
    train_total = count_split(MERGED_DIR, "train")
    val_total   = count_split(MERGED_DIR, "val")
    print(f"\n合并完成: train={train_total}  val={val_total}  总计={train_total+val_total}")

    # 生成 dataset.yaml
    yaml_path = MERGED_DIR / "dataset.yaml"
    names_lines = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    yaml_content = f"""# YOLOv8 obstacle detection - merged dataset
path: {MERGED_DIR.as_posix()}
train: images/train
val:   images/val

nc: {len(CLASS_NAMES)}
names:
{names_lines}
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"dataset.yaml 已写入: {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────────────────────

def train(yaml_path: Path, epochs: int, imgsz: int, batch: int,
          model_size: str, device: str, pretrained: str = ""):
    """调用 YOLOv8 detect 训练。"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 未安装 ultralytics，请执行: pip install ultralytics")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 支持从已有检查点继续训练
    if pretrained and Path(pretrained).exists():
        base_model = pretrained
        print(f"\n使用已有权重继续训练: {base_model}")
    else:
        base_model = f"yolov8{model_size}.pt"
        print(f"\n从预训练模型开始训练: {base_model}")

    print(f"{'='*55}")
    print(f"YOLOv8 障碍物检测训练（合并数据集）")
    print(f"  数据集:   {yaml_path}")
    print(f"  epochs:   {epochs}  imgsz: {imgsz}  batch: {batch}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'='*55}\n")

    model = YOLO(base_model)
    model.train(
        data=str(yaml_path),
        task="detect",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device if device else None,
        project=str(OUTPUT_DIR.parent),
        name="obstacle_merged",
        exist_ok=True,
        patience=20,
        # 数据增强
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.1,
        fliplr=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        mosaic=1.0,
        mixup=0.1,
    )

    best_pt = OUTPUT_DIR / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n✓ 训练完成！最佳模型: {best_pt}")
        print(f"  请更新 config.py 中 yolo_model_path 指向此路径")
    else:
        print(f"\n训练完成，未找到 best.pt，请检查: {OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# 统计新旧数据比例（供参考）
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_stats():
    """打印原始/新增数据量，帮助判断过采样倍数。"""
    orig_train = count_split(ORIG_YOLO_DIR, "train") if (ORIG_YOLO_DIR / "images" / "train").exists() else 0
    orig_val   = count_split(ORIG_YOLO_DIR, "val")   if (ORIG_YOLO_DIR / "images" / "val").exists()   else 0

    new_total = 0
    print("\n── 数据量统计 ──")
    print(f"  原始 obstacle_yolo: train={orig_train}  val={orig_val}")
    for data_dir, _ in NEW_DATA_SOURCES:
        if data_dir.exists():
            n = len(find_image_label_pairs(data_dir))
            new_total += n
            print(f"  新增 {data_dir.name}: {n} 张")

    # 台阶（跨目录）
    if STAIRS_IMG_DIR.exists() and STAIRS_LBL_DIR.exists():
        n = len(find_cross_dir_pairs(STAIRS_IMG_DIR, STAIRS_LBL_DIR))
        new_total += n
        print(f"  新增 台阶(电线杆+台阶/台阶): {n} 张 [多边形→bbox]")

    # 自行车（跨目录）
    if BICYCLE_IMG_DIR.exists() and BICYCLE_LBL_DIR.exists():
        n = len(find_cross_dir_pairs(BICYCLE_IMG_DIR, BICYCLE_LBL_DIR))
        new_total += n
        print(f"  新增 自行车(bicycle/bikes): {n} 张")

    print(f"  新增合计: {new_total} 张")

    if new_total > 0 and orig_train > 0:
        ratio = orig_train / new_total
        suggested = max(3, min(10, round(ratio)))
        print(f"\n  建议过采样倍数: {suggested}x  "
              f"（原始train/新数据 ≈ {ratio:.1f}:1，使新数据占比约 {100/(1+ratio/suggested):.0f}%）")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="障碍物数据集合并 + YOLOv8 强化训练")
    parser.add_argument("--skip-build", action="store_true",
                        help="跳过数据集合并，直接训练（需已有 training_data/obstacle_merged/）")
    parser.add_argument("--repeat",     type=int,   default=5,
                        help="新数据过采样倍数（默认5，即新数据在train中复制5份）")
    parser.add_argument("--val-ratio",  type=float, default=0.15,
                        help="新数据中验证集比例（默认0.15）")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--batch",      type=int,   default=8)
    parser.add_argument("--model-size", type=str,   default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 模型规模（默认 nano）")
    parser.add_argument("--device",     type=str,   default="",
                        help="训练设备，如 0(GPU0), cpu，留空自动选择")
    parser.add_argument("--pretrained", type=str,   default="",
                        help="从已有 .pt 权重继续训练（可选）")
    parser.add_argument("--stats-only", action="store_true",
                        help="仅打印数据量统计，不执行训练")
    parser.add_argument("--skip-train", action="store_true",
                        help="只构建合并数据集，不启动训练（用于打包上传云端）")
    args = parser.parse_args()

    os.chdir(ROOT)

    if args.stats_only:
        print_dataset_stats()
        return

    print_dataset_stats()

    yaml_path = MERGED_DIR / "dataset.yaml"

    if not args.skip_build:
        print("── 步骤 1: 合并数据集 ──")
        yaml_path = build_merged_dataset(
            repeat_times=args.repeat,
            val_ratio=args.val_ratio,
        )
    else:
        print("── 跳过数据集合并 ──")
        if not yaml_path.exists():
            print(f"[ERROR] dataset.yaml 不存在: {yaml_path}")
            sys.exit(1)

    if args.skip_train:
        print("\n── 数据集构建完成，跳过训练（用于打包上传） ──")
        print(f"数据集路径: {MERGED_DIR}")
        print("接下来运行: python pack_for_modelscope.py")
        return

    print("\n── 步骤 2: 开始训练 ──")
    train(
        yaml_path=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_size=args.model_size,
        device=args.device,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
