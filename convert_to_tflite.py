"""
将三个模型转换为TFLite格式，用于Android部署
策略：已有ONNX文件 → 用onnx2tf Python API → TFLite
"""
import os, sys, glob, shutil

OUTPUT_DIR = "training_checkpoints"


def onnx_to_tflite(onnx_path, tflite_dst, tmp_dir):
    """使用 onnx2tf Python API 将 ONNX 转换为 TFLite"""
    import onnx2tf
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=tmp_dir,
            not_use_onnxsim=False,
            non_verbose=True,
        )
    except Exception as e:
        print(f"  onnx2tf convert error: {e}")

    # 找输出的 tflite 文件
    tflite_files = glob.glob(os.path.join(tmp_dir, "**/*.tflite"), recursive=True)
    if tflite_files:
        shutil.copy(tflite_files[0], tflite_dst)
        size = os.path.getsize(tflite_dst) / 1024 / 1024
        print(f"[OK] {tflite_dst}  ({size:.1f}MB)")
        return True
    else:
        print(f"[FAIL] 未找到 tflite 文件，检查 {tmp_dir}")
        return False


# ─── 1. YOLOv8 障碍物 ───────────────────────────────────────────────────────
def convert_yolov8():
    print("\n========== [1/3] YOLOv8 障碍物模型 ==========")
    onnx_path = os.path.join(OUTPUT_DIR, "obstacle/best.onnx")
    dst = os.path.join(OUTPUT_DIR, "obstacle_best.tflite")

    # 如果 ONNX 还不存在，先用 ultralytics 导出
    if not os.path.exists(onnx_path):
        pt_path = os.path.join(OUTPUT_DIR, "obstacle/best.pt")
        if not os.path.exists(pt_path):
            print(f"[SKIP] 找不到 {pt_path}")
            return
        try:
            from ultralytics import YOLO
            model = YOLO(pt_path)
            model.export(format="onnx", imgsz=320, opset=12)
            print(f"ONNX 导出完成: {onnx_path}")
        except Exception as e:
            print(f"[ERROR] ONNX 导出失败: {e}")
            return

    if not os.path.exists(onnx_path):
        print(f"[SKIP] ONNX 文件不存在: {onnx_path}")
        return

    print(f"使用已有 ONNX: {onnx_path}  ({os.path.getsize(onnx_path)/1024/1024:.1f}MB)")
    onnx_to_tflite(onnx_path, dst, os.path.join(OUTPUT_DIR, "obstacle_onnx2tf"))


# ─── 2. 水坑 U-Net ────────────────────────────────────────────────────────────
def convert_water_unet():
    print("\n========== [2/3] 水坑 U-Net 模型 ==========")
    onnx_path = os.path.join(OUTPUT_DIR, "water_model.onnx")
    dst = os.path.join(OUTPUT_DIR, "water_model.tflite")

    if not os.path.exists(onnx_path):
        # 重新从 .pth 导出 ONNX
        pth_path = os.path.join(OUTPUT_DIR, "water_model_best.pth")
        if not os.path.exists(pth_path):
            print(f"[SKIP] 找不到 {pth_path}")
            return
        try:
            import torch, torch.nn as nn
            class UNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    def cb(i, o):
                        return nn.Sequential(
                            nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                            nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
                    def up(i, o): return nn.ConvTranspose2d(i,o,2,stride=2)
                    self.enc1=cb(3,64); self.enc2=cb(64,128)
                    self.enc3=cb(128,256); self.enc4=cb(256,512)
                    self.pool=nn.MaxPool2d(2)
                    self.bottleneck=cb(512,1024)
                    self.up4=up(1024,512); self.dec4=cb(1024,512)
                    self.up3=up(512,256);  self.dec3=cb(512,256)
                    self.up2=up(256,128);  self.dec2=cb(256,128)
                    self.up1=up(128,64);   self.dec1=cb(128,64)
                    self.out=nn.Conv2d(64,1,1); self.sigmoid=nn.Sigmoid()
                def forward(self, x):
                    e1=self.enc1(x); e2=self.enc2(self.pool(e1))
                    e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
                    b=self.bottleneck(self.pool(e4))
                    d4=self.dec4(torch.cat([self.up4(b),e4],1))
                    d3=self.dec3(torch.cat([self.up3(d4),e3],1))
                    d2=self.dec2(torch.cat([self.up2(d3),e2],1))
                    d1=self.dec1(torch.cat([self.up1(d2),e1],1))
                    return self.sigmoid(self.out(d1))

            model = UNet().cpu()
            ckpt = torch.load(pth_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
            model.eval()

            dummy = torch.randn(1, 3, 256, 256)
            torch.onnx.export(model, dummy, onnx_path,
                              opset_version=12,
                              input_names=["input"], output_names=["output"])
            print(f"ONNX 导出: {onnx_path}")
        except Exception as e:
            print(f"[ERROR] ONNX 导出失败: {e}")
            return

    print(f"使用 ONNX: {onnx_path}  ({os.path.getsize(onnx_path)/1024/1024:.1f}MB)")
    onnx_to_tflite(onnx_path, dst, os.path.join(OUTPUT_DIR, "water_onnx2tf"))


# ─── 3. 鞋型 ResNet50 ─────────────────────────────────────────────────────────
def convert_shoe():
    print("\n========== [3/3] 鞋型 ResNet50 模型 ==========")
    onnx_path = os.path.join(OUTPUT_DIR, "shoe_model.onnx")
    dst = os.path.join(OUTPUT_DIR, "shoe_model.tflite")

    if not os.path.exists(onnx_path):
        pth_path = os.path.join(OUTPUT_DIR, "shoe_model_best.pth")
        if not os.path.exists(pth_path):
            print(f"[SKIP] 找不到 {pth_path}")
            return
        try:
            import torch, torch.nn as nn, torchvision.models as models
            class ShoeResNet50(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.resnet = models.resnet50(weights=None)
                    nf = self.resnet.fc.in_features
                    self.resnet.fc = nn.Sequential(
                        nn.Dropout(0.5), nn.Linear(nf,256), nn.ReLU(),
                        nn.Dropout(0.3), nn.Linear(256,5))
                def forward(self, x): return self.resnet(x)

            model = ShoeResNet50().cpu()
            ckpt = torch.load(pth_path, map_location="cpu")
            if isinstance(ckpt, dict):
                key = "state_dict" if "state_dict" in ckpt else \
                      "model_state_dict" if "model_state_dict" in ckpt else None
                if key: model.load_state_dict(ckpt[key])
                else: model.load_state_dict(ckpt)
            model.eval()

            dummy = torch.randn(1, 3, 224, 224)
            torch.onnx.export(model, dummy, onnx_path,
                              opset_version=12,
                              input_names=["input"], output_names=["output"])
            print(f"ONNX 导出: {onnx_path}")
        except Exception as e:
            print(f"[ERROR] ONNX 导出失败: {e}")
            return

    print(f"使用 ONNX: {onnx_path}  ({os.path.getsize(onnx_path)/1024/1024:.1f}MB)")
    onnx_to_tflite(onnx_path, dst, os.path.join(OUTPUT_DIR, "shoe_onnx2tf"))


if __name__ == "__main__":
    print("开始模型转换 ONNX → TFLite (使用 onnx2tf Python API)")
    convert_yolov8()
    convert_water_unet()
    convert_shoe()

    print("\n========== 完成 ==========")
    for f in glob.glob(os.path.join(OUTPUT_DIR, "*.tflite")):
        print(f"  {f}  ({os.path.getsize(f)/1024/1024:.1f}MB)")
