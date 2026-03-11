"""
下载路面积水/水坑图片
由于网络限制，改用提示用户手动下载的方式
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, '.')

def main():
    """主函数"""
    print("=" * 60)
    print("路面积水图片下载指南")
    print("=" * 60)
    
    save_dir = Path("training_data/water_real/images")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n由于网络限制，无法自动下载。请手动操作：")
    print(f"\n方法1 - 百度图片搜索：")
    print(f"  1. 访问 https://image.baidu.com")
    print(f"  2. 搜索以下关键词：")
    print(f"     - 路面积水")
    print(f"     - 雨后水坑")
    print(f"     - 街道积水")
    print(f"     - 路面水洼")
    print(f"     - road puddle")
    print(f"  3. 下载 50~100 张清晰的积水照片")
    print(f"  4. 保存到: {save_dir}")
    
    print(f"\n方法2 - 自己拍摄（推荐）：")
    print(f"  用手机拍摄真实水坑，不同角度、光线、大小")
    print(f"  保存到: {save_dir}")
    
    print(f"\n方法3 - 千库网/摄图网：")
    print(f"  访问 https://588ku.com 或 https://699pic.com")
    print(f"  搜索'路面积水'、'雨后街道'")
    
    print(f"\n{'=' * 60}")
    print(f"下载完成后，运行标注工具生成掩码：")
    print(f"  python batch_annotate_masks.py")
    print(f"{'=' * 60}")
    
    # 统计当前图片数量
    imgs = list(save_dir.glob("*.jpg")) + list(save_dir.glob("*.png"))
    print(f"\n当前已有图片: {len(imgs)} 张")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
