"""
水坑数据集全量合并脚本
将三个来源的数据全部合并为统一的 YOLO 格式数据集：

1. water/         : 79张，已有 YOLO 标注 → 直接复制
2. water_real/    : 727张图 + 707个分割 mask → mask 转 YOLO bbox
3. Roboflow 坑洼  : 2910张，Pascal VOC XML → VOC 转 YOLO bbox

输出目录: training_data/water_merged/
  images/train/   images/val/
  labels/train/   labels/val/
  dataset.yaml

分割比例: 训练 85%  验证 15%
类别: 0=puddle（统一，水坑/坑洼均归为此类）
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("[警告] Pillow 未安装，mask 转换将用 numpy；若 numpy 也没有则跳过 water_real")

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

# ==================== 路径配置 ====================
BASE = Path(r"d:\bujiwbu zhong\-- 不积洼步_14日改\-- 不积洼步_23日改\不积洼步_1.23\不积洼步(1)\不积洼步\training_data")

SRC_WATER      = BASE / "water" / "water_puddle_detector"   # 已有 YOLO 标注
SRC_WATER_REAL = BASE / "water_real"                         # mask 目录
SRC_VOC        = BASE / "-- test train valid README.dataset.txt README.roboflow.txt"  # Roboflow VOC

DST = BASE / "water_merged"

TRAIN_RATIO = 0.85   # 训练集占比
CLASS_ID    = 0      # 全部归为 puddle
SEED        = 42

# ==================== 工具函数 ====================

def mkdir(p):
    p.mkdir(parents=True, exist_ok=True)

def img_exts():
    return {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def bbox_from_mask_pil(mask_path):
    """用 Pillow 从二值 mask 提取外接矩形，返回归一化 (cx,cy,bw,bh) 或 None"""
    img = Image.open(mask_path).convert('L')
    w, h = img.size
    pixels = img.load()
    xs, ys = [], []
    for y in range(h):
        for x in range(w):
            if pixels[x, y] > 127:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    cx = (xmin + xmax) / 2 / w
    cy = (ymin + ymax) / 2 / h
    if bw < 0.001 or bh < 0.001:
        return None
    return cx, cy, bw, bh

def bbox_from_mask_numpy(mask_path):
    """用 numpy 从二值 mask 提取外接矩形"""
    from PIL import Image
    img = Image.open(mask_path).convert('L')
    arr = np.array(img)
    ys, xs = np.where(arr > 127)
    if len(xs) == 0:
        return None
    h, w = arr.shape
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    cx = (xmin + xmax) / 2 / w
    cy = (ymin + ymax) / 2 / h
    if bw < 0.001 or bh < 0.001:
        return None
    return cx, cy, bw, bh

def bbox_from_mask(mask_path):
    if PIL_OK:
        return bbox_from_mask_numpy(mask_path) if NUMPY_OK else bbox_from_mask_pil(mask_path)
    return None

def voc_xml_to_yolo(xml_path):
    """解析 VOC XML，返回 YOLO 格式行列表"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None
    size = root.find("size")
    if size is None:
        return None
    width  = int(size.find("width").text)
    height = int(size.find("height").text)
    if width == 0 or height == 0:
        return None
    lines = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.find("xmin").text)
        xmax = float(bb.find("xmax").text)
        ymin = float(bb.find("ymin").text)
        ymax = float(bb.find("ymax").text)
        cx = max(0.0, min(1.0, (xmin + xmax) / 2 / width))
        cy = max(0.0, min(1.0, (ymin + ymax) / 2 / height))
        bw = max(0.0, min(1.0, (xmax - xmin) / width))
        bh = max(0.0, min(1.0, (ymax - ymin) / height))
        if bw > 0 and bh > 0:
            lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines

def split_and_copy(pairs, dst_img_train, dst_img_val, dst_lbl_train, dst_lbl_val):
    """pairs = [(img_path, label_lines_or_txt_path), ...]"""
    random.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_RATIO)
    results = {'train': 0, 'val': 0}
    for i, (img_path, label) in enumerate(pairs):
        split = 'train' if i < n_train else 'val'
        dst_img_dir = dst_img_train if split == 'train' else dst_img_val
        dst_lbl_dir = dst_lbl_train if split == 'train' else dst_lbl_val

        # 目标文件名（去空格避免 YOLO 解析问题）
        safe_name = Path(img_path).name.replace(' ', '_')
        dst_img = dst_img_dir / safe_name
        dst_lbl = dst_lbl_dir / (Path(safe_name).stem + '.txt')

        # 复制图片
        shutil.copy2(img_path, dst_img)

        # 写标注
        if isinstance(label, (str, Path)):
            shutil.copy2(label, dst_lbl)
        else:
            with open(dst_lbl, 'w', encoding='utf-8') as f:
                f.write('\n'.join(label) + '\n')

        results[split] += 1
    return results

# ==================== 主流程 ====================

def main():
    random.seed(SEED)

    # 建目标目录
    dst_img_train = DST / "images" / "train"
    dst_img_val   = DST / "images" / "val"
    dst_lbl_train = DST / "labels" / "train"
    dst_lbl_val   = DST / "labels" / "val"
    for d in [dst_img_train, dst_img_val, dst_lbl_train, dst_lbl_val]:
        mkdir(d)

    print("=" * 65)
    print("水坑数据集全量合并")
    print("=" * 65)

    all_pairs = []

    # ── 来源 1：water/ 已有 YOLO 标注 ──────────────────────────────
    print("\n[1/3] 处理 water/（已有 YOLO 标注）...")
    # 图片散落在 water/ 下（直接放在 water/ 目录），标注在 water_puddle_detector/labels/
    lbl_dir = SRC_WATER / "labels"
    water_root = BASE / "water"
    # 先建一个 stem -> 图片路径 的索引（递归搜索所有子目录）
    img_index = {}
    for ext in img_exts():
        for p in water_root.rglob(f"*{ext}"):
            img_index[p.stem] = p
    pairs1 = []
    for lbl in lbl_dir.glob("*.txt"):
        stem = lbl.stem
        if stem in img_index:
            pairs1.append((img_index[stem], lbl))
    print(f"  找到 {len(pairs1)} 对（图片+标注）")

    # 直接加入总池
    for img_path, lbl_path in pairs1:
        all_pairs.append((img_path, lbl_path))

    # ── 来源 2：water_real/ mask → YOLO bbox ──────────────────────
    print("\n[2/3] 处理 water_real/（mask → YOLO bbox）...")
    if not (PIL_OK or NUMPY_OK):
        print("  [跳过] 需要 Pillow 或 numpy，请运行: pip install pillow numpy")
    else:
        real_img_dir  = SRC_WATER_REAL / "images"
        real_mask_dir = SRC_WATER_REAL / "masks"
        pairs2 = []
        skipped = 0
        for mask_p in real_mask_dir.glob("*.png"):
            stem = mask_p.stem
            # 找对应图片（jpg/jpeg/png）
            img_p = None
            for ext in img_exts():
                c = real_img_dir / (stem + ext)
                if c.exists():
                    img_p = c
                    break
            if img_p is None:
                skipped += 1
                continue
            bbox = bbox_from_mask(mask_p)
            if bbox is None:
                skipped += 1
                continue
            cx, cy, bw, bh = bbox
            label_lines = [f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
            pairs2.append((img_p, label_lines))
        print(f"  转换成功: {len(pairs2)} 张  |  跳过（无mask/无图）: {skipped}")
        all_pairs.extend(pairs2)

    # ── 来源 3：Roboflow VOC XML → YOLO bbox ──────────────────────
    print("\n[3/3] 处理 Roboflow 坑洼数据集（VOC XML → YOLO bbox）...")
    pairs3 = []
    skipped3 = 0
    for split_name in ["train", "valid", "test"]:
        split_dir = SRC_VOC / split_name
        if not split_dir.exists():
            continue
        for xml_p in split_dir.glob("*.xml"):
            # 找图片
            img_p = xml_p.with_suffix(".jpg")
            if not img_p.exists():
                img_p = xml_p.with_suffix(".png")
            if not img_p.exists():
                skipped3 += 1
                continue
            lines = voc_xml_to_yolo(xml_p)
            if not lines:
                skipped3 += 1
                continue
            # 加前缀避免文件名冲突
            pairs3.append((img_p, lines, f"pothole_{split_name}_"))
    print(f"  转换成功: {len(pairs3)} 张  |  跳过: {skipped3}")

    # Roboflow 数据加前缀后加入 all_pairs
    for img_p, lines, prefix in pairs3:
        all_pairs.append((img_p, lines, prefix))

    # ── 分割并复制到目标目录 ──────────────────────────────────────
    print(f"\n汇总共 {len(all_pairs)} 条数据，开始按 {TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%} 分割...")

    random.shuffle(all_pairs)
    n_train = int(len(all_pairs) * TRAIN_RATIO)
    cnt = {'train': 0, 'val': 0}

    for i, item in enumerate(all_pairs):
        if len(item) == 3:
            img_path, label, prefix = item
        else:
            img_path, label = item
            prefix = ""

        split = 'train' if i < n_train else 'val'
        dst_img_dir = dst_img_train if split == 'train' else dst_img_val
        dst_lbl_dir = dst_lbl_train if split == 'train' else dst_lbl_val

        safe_name = prefix + Path(img_path).name.replace(' ', '_')
        dst_img = dst_img_dir / safe_name
        dst_lbl = dst_lbl_dir / (Path(safe_name).stem + '.txt')

        # 如果目标已存在（来源1 可能重名），加序号
        if dst_img.exists():
            stem = Path(safe_name).stem
            ext  = Path(safe_name).suffix
            safe_name = f"{stem}_{i}{ext}"
            dst_img = dst_img_dir / safe_name
            dst_lbl = dst_lbl_dir / (Path(safe_name).stem + '.txt')

        shutil.copy2(img_path, dst_img)

        if isinstance(label, (str, Path)):
            shutil.copy2(label, dst_lbl)
        else:
            with open(dst_lbl, 'w', encoding='utf-8') as f:
                f.write('\n'.join(label) + '\n')

        cnt[split] += 1

    print(f"  train: {cnt['train']} 张  |  val: {cnt['val']} 张")

    # ── 生成 dataset.yaml ──────────────────────────────────────────
    yaml_path = DST / "dataset.yaml"
    yaml_content = f"""# 水坑检测合并数据集
# 来源: water(79) + water_real(mask转换) + Roboflow坑洼(VOC转换)
# 生成工具: merge_water_datasets.py

path: {DST.as_posix()}
train: images/train
val:   images/val

nc: 1
names:
  0: puddle
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n已写入 dataset.yaml: {yaml_path}")

    # ── 最终统计 ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("完成！最终数据集统计：")
    for split in ['train', 'val']:
        n_img = len(list((DST / "images" / split).glob("*")))
        n_lbl = len(list((DST / "labels" / split).glob("*.txt")))
        print(f"  {split:5s}: {n_img} 图片  {n_lbl} 标注")
    print(f"\n数据集路径: {DST}")
    print(f"YAML 配置 : {yaml_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
