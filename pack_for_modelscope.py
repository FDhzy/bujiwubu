"""
打包脚本：将合并好的数据集打包为 upload_to_modelscope.zip
用法：
  python pack_for_modelscope.py            # 默认打包障碍物数据集
  python pack_for_modelscope.py water      # 打包水坑数据集
  python pack_for_modelscope.py obstacle   # 打包障碍物数据集
"""
import zipfile
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# 根据参数选择数据集
MODE = sys.argv[1].lower() if len(sys.argv) > 1 else "obstacle"
if MODE == "water":
    MERGED_DIR = ROOT / "training_data" / "water_merged"
    OUTPUT_ZIP = ROOT / "upload_to_modelscope.zip"
else:
    MERGED_DIR = ROOT / "training_data" / "obstacle_merged"
    OUTPUT_ZIP = ROOT / "upload_to_modelscope.zip"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*") if f.suffix.lower() in IMG_EXTS)


def main():
    if not MERGED_DIR.exists():
        print("[ERROR] 合并数据集目录不存在，请先运行：")
        print("  python merge_and_train_obstacle.py --repeat 10 --skip-train")
        sys.exit(1)

    train_n = count_images(MERGED_DIR / "images" / "train")
    val_n   = count_images(MERGED_DIR / "images" / "val")
    print(f"数据集统计: train={train_n}  val={val_n}  合计={train_n+val_n}")

    if train_n == 0:
        print("[ERROR] train 目录为空，请先构建合并数据集")
        sys.exit(1)

    print(f"\n打包中: {MERGED_DIR} -> {OUTPUT_ZIP}")

    file_count = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path in sorted(MERGED_DIR.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(ROOT)  # 保留 training_data/obstacle_merged/... 路径
                zf.write(file_path, arcname)
                file_count += 1
                if file_count % 500 == 0:
                    print(f"  已添加 {file_count} 个文件...")

    size_mb = OUTPUT_ZIP.stat().st_size / 1024 ** 2
    print(f"\n打包完成: {OUTPUT_ZIP}")
    print(f"  文件数: {file_count}")
    print(f"  大小:   {size_mb:.1f} MB")
    print(f"\n请将 {OUTPUT_ZIP.name} 上传到魔搭社区 Notebook，")
    print(f"然后打开 modelscope_train.ipynb 按顺序运行即可。")


if __name__ == "__main__":
    main()
