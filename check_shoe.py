"""鞋型识别模块全面检查"""
import os, sys
from utils.logger import setup_logger
import logging; setup_logger(level=logging.WARNING)

print("=== 鞋型识别模块检查 ===\n")

# 1. 检查模型文件（优先使用魔搭训练的本地权重）
model_path = os.path.join('training_checkpoints', 'checkpoints', 'shoe_best.pth')
if not os.path.exists(model_path):
    model_path = os.path.join('assets', 'models', 'cv_resnest101_general_recognition', 'pytorch_model.pt')
print(f"[1] PyTorch模型文件: {model_path}")
print(f"    存在: {os.path.exists(model_path)}")

# 2. 检查数据库
db_path = os.path.join('assets', 'models', 'shoe_recognition.db')
print(f"[2] 数据库文件: {db_path}")
print(f"    存在: {os.path.exists(db_path)}")
if os.path.exists(db_path):
    print(f"    大小: {os.path.getsize(db_path)/1024:.1f} KB")

# 3. 检查依赖
print("\n[3] 依赖检查:")
for pkg, imp in [("cv2","cv2"),("torch","torch"),("torchvision","torchvision"),("PIL","PIL")]:
    try:
        m = __import__(imp)
        ver = getattr(m, '__version__', 'OK')
        print(f"    {pkg}: OK ({ver})")
    except ImportError:
        print(f"    {pkg}: MISSING")

# 4. 测试ShoeAnalyzer初始化
print("\n[4] ShoeAnalyzer 初始化:")
try:
    from models.shoe_analyzer import ShoeAnalyzer, ShoeType
    analyzer = ShoeAnalyzer(language='zh')
    print("    ShoeAnalyzer: OK (CV模式)")
    print(f"    ORB特征提取: {analyzer.orb is not None}")
    print(f"    数据库连接: {analyzer.database is not None}")
    print(f"    人脸检测器: {analyzer.face_cascade is not None}")
except Exception as e:
    print(f"    ShoeAnalyzer: FAILED - {e}")

# 5. 测试PyTorchShoeModel
print("\n[5] PyTorchShoeModel:")
try:
    from models.pytorch_shoe_model import PyTorchShoeModel
    m = PyTorchShoeModel(use_database=True)
    if os.path.exists(model_path):
        ok = m.load_model(model_path)
        status = "OK" if ok else "FAILED"
        print(f"    模型加载: {status}")
    else:
        print("    模型文件不存在 -> 回退到CV识别模式")
        print("    （这是正常情况，CV识别仍可用）")
except Exception as e:
    print(f"    PyTorchShoeModel: FAILED - {e}")

# 6. 测试shoe_setup_screen 核心逻辑
print("\n[6] shoe_setup_screen 导入检查:")
try:
    from ui.shoe_setup_screen import ShoeSetupScreen, SHOE_TYPE_NAMES
    print(f"    ShoeSetupScreen: OK")
    print(f"    SHOE_TYPE_NAMES: {SHOE_TYPE_NAMES}")
except Exception as e:
    print(f"    FAILED: {e}")

# 7. 测试shoe_type_choice_screen
print("\n[7] shoe_type_choice_screen 导入检查:")
try:
    from ui.shoe_type_choice_screen import ShoeTypeChoiceScreen
    print("    ShoeTypeChoiceScreen: OK")
except Exception as e:
    print(f"    FAILED: {e}")

# 8. 测试CV分析流程（含结果完整性）
print("\n[8] CV分析流程测试:")
try:
    import numpy as np
    from models.shoe_analyzer import ShoeAnalyzer
    analyzer = ShoeAnalyzer(language='zh')
    test_img = np.ones((480, 640, 3), dtype='uint8') * 100
    result = analyzer.analyze(test_img)
    print(f"    shoe_type: {result.shoe_type.value}")
    print(f"    confidence: {result.confidence:.2f}")
    print(f"    safe_puddle_depth_cm: {result.safe_puddle_depth_cm}")
    print(f"    warning_threshold: {result.puddle_warning_threshold:.4f}")
    print(f"    description: {result.description}")
except Exception as e:
    import traceback
    print("    CV分析: FAILED")
    traceback.print_exc()

# 9. 检查config中鞋型相关字段
print("\n[9] Config 鞋型配置:")
try:
    from utils.config import get_config
    cfg = get_config()
    print(f"    require_shoe_analysis: {cfg.shoe.require_shoe_analysis}")
    print(f"    last_shoe_type: {cfg.last_shoe_type}")
    print(f"    shoe_analysis_complete: {cfg.shoe_analysis_complete}")
    print(f"    puddle_warning_threshold: {cfg.puddle_warning_threshold}")
except Exception as e:
    print(f"    FAILED: {e}")

print("\n=== 检查完成 ===")
