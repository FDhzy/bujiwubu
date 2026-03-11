"""
从网络收集真实水坑图片
使用免费图库API下载
"""
import os
import urllib.request
import json
from pathlib import Path
import time


def download_from_unsplash(query="puddle", num_images=50):
    """从Unsplash下载水坑图片"""
    print(f"搜索Unsplash: {query}")
    
    # Unsplash Source API (无需认证)
    base_url = f"https://source.unsplash.com/featured/?{query}"
    
    output_dir = Path("training_data/water_real/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"下载 {num_images} 张图片到 {output_dir}")
    print("-" * 60)
    
    downloaded = 0
    for i in range(num_images):
        try:
            # 添加随机参数避免缓存
            url = f"{base_url}&sig={i}"
            output_path = output_dir / f"unsplash_{query}_{i:03d}.jpg"
            
            # 下载图片
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                with open(output_path, 'wb') as f:
                    f.write(response.read())
            
            downloaded += 1
            print(f"  已下载: {downloaded}/{num_images}", end='\r')
            
            # 延迟避免请求过快
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\n  下载失败 {i}: {e}")
            continue
    
    print(f"\n✓ 下载完成: {downloaded} 张图片")
    return downloaded


def download_water_related_images():
    """下载多种水相关图片"""
    print("=" * 60)
    print("从网络收集真实水坑/积水图片")
    print("=" * 60)
    
    # 搜索关键词列表
    queries = [
        "puddle",
        "water reflection ground",
        "rain puddle",
        "street puddle",
        "wet ground",
    ]
    
    total_downloaded = 0
    
    for query in queries:
        print(f"\n搜索关键词: {query}")
        count = download_from_unsplash(query, num_images=20)
        total_downloaded += count
        time.sleep(1)  # 关键词间延迟
    
    print("\n" + "=" * 60)
    print(f"总计下载: {total_downloaded} 张图片")
    print("=" * 60)
    
    # 提示用户标注
    print("\n下一步:")
    print("1. 检查下载的图片质量")
    print("2. 删除不相关的图片")
    print("3. 使用 water_data_collector.py 标注水坑区域")
    print("   python water_data_collector.py --mode folder --source training_data/water_real/images")


if __name__ == "__main__":
    try:
        download_water_related_images()
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
    except Exception as e:
        print(f"\n错误: {e}")
