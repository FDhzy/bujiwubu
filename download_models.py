"""
下载 ML 模型脚本
用于下载盲人导航助手所需的 TFLite 模型文件
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
    },
    "yolov8n.tflite": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-oiv7.pt",
        "description": "YOLOv8-Nano 障碍物检测模型 (需要转换为 TFLite)",
        "size_mb": 6.2,
        "note": "这是 PyTorch 格式，需要转换为 TFLite"
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
    """下载所有模型文件"""
    print("=" * 60)
    print("盲人导航助手 - ML 模型下载工具")
    print("=" * 60)
    
    # 确保模型目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"\n模型将保存到: {MODEL_DIR}")
    
    # 检查现有模型
    print("\n" + "-" * 60)
    print("检查现有模型...")
    existing = []
    missing = []
    
    for model_name, info in MODELS.items():
        model_path = os.path.join(MODEL_DIR, model_name)
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / 1024 / 1024
            print(f"  ✓ {model_name} (已存在, {size:.1f} MB)")
            existing.append(model_name)
        else:
            print(f"  ✗ {model_name} (缺失)")
            missing.append((model_name, info))
    
    if not missing:
        print("\n✓ 所有模型都已存在，无需下载!")
        return
    
    # 下载缺失的模型
    print("\n" + "-" * 60)
    print(f"需要下载 {len(missing)} 个模型")
    
    for model_name, info in missing:
        print(f"\n{'=' * 60}")
        print(f"模型: {model_name}")
        print(f"描述: {info['description']}")
        print(f"预计大小: {info['size_mb']} MB")
        if 'note' in info:
            print(f"注意: {info['note']}")
        
        model_path = os.path.join(MODEL_DIR, model_name)
        
        # 询问用户
        response = input("\n是否下载此模型? (y/n): ").strip().lower()
        if response != 'y':
            print("跳过此模型")
            continue
        
        # 下载
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
    all_exist = True
    for model_name in MODELS.keys():
        model_path = os.path.join(MODEL_DIR, model_name)
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / 1024 / 1024
            print(f"  ✓ {model_name} ({size:.1f} MB)")
        else:
            print(f"  ✗ {model_name} (缺失)")
            all_exist = False
    
    if all_exist:
        print("\n✓ 所有模型已就绪! 可以运行 main.py 启动应用")
    else:
        print("\n⚠ 部分模型缺失，应用将使用 CV 回退方案")
        print("  如需完整功能，请重新运行此脚本下载缺失模型")


if __name__ == "__main__":
    download_models()
