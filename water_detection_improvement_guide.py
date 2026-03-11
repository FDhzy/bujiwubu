"""
水坑检测改进完整指南
包含所有工具和使用方法
"""

GUIDE = """
╔══════════════════════════════════════════════════════════════╗
║           水坑检测改进完整方案 - 不积洼步                    ║
╚══════════════════════════════════════════════════════════════╝

【当前状态】
✓ 合成数据集: 500张多样化水坑图片
✓ 训练模型: U-Net架构，355MB
✓ 检测器: PyTorch实现 + 水深估算

【数据收集选项】

1. 摄像头实时收集 (推荐)
   命令: python water_data_collector.py --mode camera
   
   操作步骤:
   - 按 'c' 捕获当前帧
   - 用鼠标绘制水坑区域
   - 按 's' 保存标注
   - 按 'q' 退出

2. 文件夹批量标注
   命令: python water_data_collector.py --mode folder --source <图片文件夹>
   
   适用场景:
   - 已有水坑照片需要标注
   - 从网络下载的图片批量处理

3. 生成更多合成数据
   命令: python generate_diverse_water_data.py
   
   生成类型:
   - 道路水坑 (沥青路面)
   - 人行道水坑 (砖块路面)
   - 泥水坑 (泥土路面)
   - 反光水坑 (强烈反光)

【模型训练】

1. 训练新模型
   命令: python train_water_model_real.py --epochs 30 --batch-size 8
   
   参数说明:
   --epochs: 训练轮数 (建议20-50)
   --batch-size: 批次大小 (根据显存调整)
   --lr: 学习率 (默认0.001)

2. 在现有模型上微调
   修改 train_water_model_real.py:
   - 加载已有模型权重
   - 使用较小学习率 (0.0001)
   - 减少训练轮数 (10-20)

【模型使用】

from models.pytorch_water_detector import PyTorchWaterDetector

# 创建检测器
detector = PyTorchWaterDetector(
    model_path="training_checkpoints/water_model_best.pth",
    threshold=0.5
)

# 检测水坑
result = detector.detect(frame)
if result:
    print(f"覆盖率: {result.coverage_ratio:.2%}")
    print(f"区域数: {result.num_regions}")
    
    # 估算水深
    depth = detector.estimate_depth(frame, result.mask)
    print(f"相对水深: {depth:.2f}")

【文件位置】

训练数据:
- training_data/water_real/images/    (图片)
- training_data/water_real/masks/     (标注)

模型文件:
- training_checkpoints/water_model_best.pth   (最佳模型)
- training_checkpoints/water_model_latest.pth (最新模型)

工具脚本:
- water_data_collector.py          (数据收集)
- generate_diverse_water_data.py   (合成数据生成)
- train_water_model_real.py        (模型训练)
- pytorch_water_detector.py        (检测器实现)

【改进建议】

1. 提高准确率:
   ✓ 收集100+张真实水坑照片
   ✓ 在真实数据上微调模型
   ✓ 调整检测阈值 (0.3-0.7)

2. 增强鲁棒性:
   ✓ 不同天气条件 (雨/晴/阴)
   ✓ 不同地面类型 (沥青/水泥/土路)
   ✓ 不同光照条件 (白天/黄昏/夜晚)

3. 性能优化:
   ✓ 使用GPU加速 (如果可用)
   ✓ 减小输入分辨率 (256x256)
   ✓ 模型量化 (TensorRT/ONNX)

【注意事项】

- 合成数据训练的模型在真实场景可能有偏差
- 建议至少收集50-100张真实照片进行微调
- 标注时注意水坑边界要准确
- 定期评估模型在真实场景的表现

═══════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(GUIDE)
