"""
自动下载 ML 模型脚本（无需交互）
"""
import os
import urllib.request
import sys

# 模型下载链接和本地路径
MODELS = {
    "deeplabv3_water.tflite": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite",
        "description": "DeepLabV3 水坑检测模型 (语义分割)",
        "size_mb": 2.8
    }
}

# 模型保存目录
MODEL_DIR = os.path.join(os.path.dirname(__file__), "assets", "models")


def download_file(url: str, dest_path: str, description: str) -> bool:
    """下载文件并显示进度"""
    print(f"\n正在下载: {description}")
    print(f"URL: {url}")
    print(f"保存到: {dest_path}")
    
    try:
        # 创建目录
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # 下载文件
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\r进度: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print("\n✓ 下载完成!")
        return True
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False


def download_models():
    """自动下载所有模型文件"""
    print("=" * 60)
    print("盲人导航助手 - ML 模型自动下载工具")
    print("=" * 60)
    
    # 确保模型目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"\n模型将保存到: {MODEL_DIR}")
    
    # 下载所有模型
    for model_name, info in MODELS.items():
        model_path = os.path.join(MODEL_DIR, model_name)
        
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / 1024 / 1024
            print(f"\n✓ {model_name} 已存在 ({size:.1f} MB)，跳过下载")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"模型: {model_name}")
        print(f"描述: {info['description']}")
        print(f"预计大小: {info['size_mb']} MB")
        
        # 自动下载
        success = download_file(info['url'], model_path, info['description'])
        
        if success:
            print(f"✓ {model_name} 下载成功!")
        else:
            print(f"✗ {model_name} 下载失败")
    
    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)
    
    # 最终检查
    print("\n模型状态:")
    for model_name in MODELS.keys():
        model_path = os.path.join(MODEL_DIR, model_name)
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / 1024 / 1024
            print(f"  ✓ {model_name} ({size:.1f} MB)")
        else:
            print(f"  ✗ {model_name} (缺失)")


if __name__ == "__main__":
    download_models()
