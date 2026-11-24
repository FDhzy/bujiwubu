"""Open a local shoe photo; offer an experimental second opinion, never save it."""
from pathlib import Path
import sys
import cv2
import numpy as np
from models.desktop_shoe import DesktopShoeSuggestion, SHOE_LABELS

ROOT = Path(__file__).resolve().parent

def inspect(path, model):
    p = Path(path)
    if p.stat().st_size > 25 * 1024 * 1024:
        raise ValueError("Photo exceeds 25 MB")
    frame = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    return model.suggest(frame)

def main():
    from tkinter import Tk, filedialog, messagebox
    root=Tk()
    root.withdraw()
    try:
        path=filedialog.askopenfilename(title="选择单只鞋的侧面照片",
                                        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        model=DesktopShoeSuggestion(ROOT/'training_checkpoints/shoe_model.tflite',ROOT/'shoe-params.json')
        label,_=inspect(path,model)
        text="证据不足，请回主界面手动选择鞋型。"
        if label is not None:
            text=f"照片建议：{SHOE_LABELS[label]}。\n请自行确认并手选。这只是实验建议，不判断鞋能涉多深的水。"
        messagebox.showinfo("鞋型实验建议",text,parent=root)
    except Exception:
        messagebox.showerror("识别未完成","请确认模型文件和图片可用，或回主界面手动选择。",parent=root)
    finally:
        root.destroy()

if __name__=="__main__":
    main()

