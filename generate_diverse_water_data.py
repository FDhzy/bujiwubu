"""
生成多样化的水坑训练数据
包含不同场景、光照、形状的水坑
"""
import numpy as np
import cv2
from pathlib import Path


def generate_road_puddle(index, output_dir):
    """生成道路水坑"""
    # 创建沥青路面背景
    bg = np.random.randint(40, 80, (256, 256, 3), dtype=np.uint8)
    
    # 添加路面纹理
    noise = np.random.randint(-15, 15, (256, 256, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 创建水坑mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    # 不规则水坑形状
    num_puddles = np.random.randint(1, 4)
    for _ in range(num_puddles):
        center = (np.random.randint(40, 216), np.random.randint(40, 216))
        axes = (np.random.randint(20, 70), np.random.randint(15, 50))
        angle = np.random.randint(0, 180)
        
        # 绘制水坑
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        
        # 添加水面反光（比路面暗但更亮）
        water_color = bg[center[1], center[0]] + np.random.randint(10, 30)
        water_color = np.clip(water_color, 0, 255)
        cv2.ellipse(bg, center, axes, angle, 0, 360, water_color.tolist(), -1)
        
        # 添加高光反射
        highlight_center = (center[0] - axes[0]//3, center[1] - axes[1]//3)
        cv2.circle(bg, highlight_center, 5, (255, 255, 255), -1)
    
    # 保存
    save_name = f"road_puddle_{index:04d}.jpg"
    cv2.imwrite(str(output_dir / "images" / save_name), bg)
    cv2.imwrite(str(output_dir / "masks" / save_name.replace('.jpg', '.png')), mask)


def generate_sidewalk_puddle(index, output_dir):
    """生成人行道水坑"""
    # 创建混凝土背景
    bg = np.random.randint(150, 200, (256, 256, 3), dtype=np.uint8)
    
    # 添加砖块纹理
    for i in range(0, 256, 32):
        cv2.line(bg, (i, 0), (i, 255), (120, 120, 120), 1)
    for j in range(0, 256, 32):
        cv2.line(bg, (0, j), (255, j), (120, 120, 120), 1)
    
    # 创建水坑
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    center = (np.random.randint(60, 196), np.random.randint(60, 196))
    axes = (np.random.randint(25, 60), np.random.randint(20, 45))
    angle = np.random.randint(0, 180)
    
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    
    # 水面颜色（灰色反光）
    water_color = (140, 145, 150)
    cv2.ellipse(bg, center, axes, angle, 0, 360, water_color, -1)
    
    # 保存
    save_name = f"sidewalk_puddle_{index:04d}.jpg"
    cv2.imwrite(str(output_dir / "images" / save_name), bg)
    cv2.imwrite(str(output_dir / "masks" / save_name.replace('.jpg', '.png')), mask)


def generate_muddy_puddle(index, output_dir):
    """生成泥水坑"""
    # 创建泥土背景
    bg = np.random.randint(80, 120, (256, 256, 3), dtype=np.uint8)
    bg[:, :, 0] = np.clip(bg[:, :, 0] + 20, 0, 255)  # 偏红色
    bg[:, :, 2] = np.clip(bg[:, :, 2] - 20, 0, 255)  # 减少蓝色
    
    # 添加泥土纹理
    for _ in range(100):
        x, y = np.random.randint(0, 256, 2)
        color = tuple(np.random.randint(60, 100, 3).tolist())
        cv2.circle(bg, (x, y), 2, color, -1)
    
    # 创建泥水坑
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    center = (np.random.randint(50, 206), np.random.randint(50, 206))
    axes = (np.random.randint(30, 80), np.random.randint(25, 60))
    angle = np.random.randint(0, 180)
    
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    
    # 泥水颜色（褐色）
    water_color = (100, 90, 70)
    cv2.ellipse(bg, center, axes, angle, 0, 360, water_color, -1)
    
    # 保存
    save_name = f"muddy_puddle_{index:04d}.jpg"
    cv2.imwrite(str(output_dir / "images" / save_name), bg)
    cv2.imwrite(str(output_dir / "masks" / save_name.replace('.jpg', '.png')), mask)


def generate_reflection_puddle(index, output_dir):
    """生成有强烈反光的水坑"""
    # 创建地面背景
    bg = np.random.randint(100, 140, (256, 256, 3), dtype=np.uint8)
    
    # 创建水坑
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    center = (np.random.randint(50, 206), np.random.randint(50, 206))
    axes = (np.random.randint(30, 70), np.random.randint(25, 55))
    angle = np.random.randint(0, 180)
    
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    
    # 深色水面
    water_color = (70, 75, 80)
    cv2.ellipse(bg, center, axes, angle, 0, 360, water_color, -1)
    
    # 添加强烈的天空反光
    for i in range(5):
        offset_x = np.random.randint(-axes[0]//2, axes[0]//2)
        offset_y = np.random.randint(-axes[1]//2, axes[1]//2)
        highlight = (center[0] + offset_x, center[1] + offset_y)
        cv2.circle(bg, highlight, np.random.randint(3, 8), (200, 210, 220), -1)
    
    # 保存
    save_name = f"reflection_puddle_{index:04d}.jpg"
    cv2.imwrite(str(output_dir / "images" / save_name), bg)
    cv2.imwrite(str(output_dir / "masks" / save_name.replace('.jpg', '.png')), mask)


def generate_diverse_dataset():
    """生成多样化的数据集"""
    print("=" * 60)
    print("生成多样化水坑数据集")
    print("=" * 60)
    
    output_dir = Path("training_data/water_real")
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    
    # 统计现有数据
    existing = len(list((output_dir / "images").glob("*.jpg")))
    print(f"\n现有数据: {existing} 张")
    
    # 生成新数据
    num_samples = 200
    print(f"生成新数据: {num_samples} 张")
    print("-" * 60)
    
    generators = [
        generate_road_puddle,
        generate_sidewalk_puddle,
        generate_muddy_puddle,
        generate_reflection_puddle,
    ]
    
    for i in range(num_samples):
        generator = generators[i % len(generators)]
        generator(existing + i, output_dir)
        
        if (i + 1) % 50 == 0:
            print(f"  已生成: {i+1}/{num_samples}")
    
    # 统计
    total = len(list((output_dir / "images").glob("*.jpg")))
    print(f"\n✓ 生成完成!")
    print(f"  总数据量: {total} 张")
    print(f"  保存位置: {output_dir}")


if __name__ == "__main__":
    generate_diverse_dataset()
