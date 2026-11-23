"""导出 YOLOv8 障碍物模型为 TFLite 格式"""
import shutil
from pathlib import Path

MODEL_PT = "training_checkpoints/obstacle/best.pt"
DEST = "assets/models/yolov8_obstacle.tflite"

try:
    from ultralytics import YOLO
    print("加载模型...")
    model = YOLO(MODEL_PT)
    
    print("开始导出 TFLite (imgsz=256)...")
    exported_path = model.export(format="tflite", imgsz=256)
    print(f"导出原始路径: {exported_path}")
    
    # 复制到 assets/models/
    src = Path(str(exported_path))
    if src.exists():
        shutil.copy2(src, DEST)
        print(f"✓ 已复制到: {DEST}")
    else:
        # 尝试查找生成的 tflite 文件
        search_dir = Path("training_checkpoints/obstacle")
        tflite_files = list(search_dir.rglob("*.tflite"))
        if tflite_files:
            shutil.copy2(tflite_files[0], DEST)
            print(f"✓ 找到并复制: {tflite_files[0]} -> {DEST}")
        else:
            print("✗ 未找到 .tflite 文件")
            
except Exception as e:
    import traceback
    print(f"✗ 导出失败:")
    traceback.print_exc()
