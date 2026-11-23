"""
语音预生成脚本
将 config 中所有固定中文短语提前合成为 mp3 文件，
保存到 assets/audio_cache/ 目录，下次启动时直接加载，
无需联网合成，大幅降低首次播报延迟。

用法:
    python prebuild_audio.py                   # 默认音色 zh-CN-XiaoxiaoNeural
    python prebuild_audio.py --voice zh-CN-YunjianNeural
    python prebuild_audio.py --force           # 强制重新生成已存在的文件
"""
import os
import sys
import asyncio
import hashlib
import argparse
from pathlib import Path

# 切到项目根目录
ROOT = Path(__file__).parent
os.chdir(ROOT)

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# ── 预生成目录 ────────────────────────────────────────────────────────────────
CACHE_DIR = ROOT / "assets" / "audio_cache"

# ── 所有需要预生成的固定短语 ──────────────────────────────────────────────────
# 来源：config.py AudioConfig.phrases_zh + 常用 UI 提示语
FIXED_PHRASES = [
    # 系统状态
    "避障助手启动中",
    "准备就绪。请将摄像头向前",
    "发生错误，请重启应用",
    "摄像头可能被遮挡",
    "电量低，切换到省电模式",
    "避障已暂停",
    "避障已恢复",

    # 导航方向
    "前方畅通",
    "请直行",
    "请向左转",
    "请稍微向左",
    "请向右转",
    "请稍微向右",
    "注意，请慢行",
    "停止！前方有危险",

    # 水坑
    "前方有水坑",
    "左侧有水坑",
    "右侧有水坑",
    "前方浅水坑，可以通过",
    "前方深水坑，请绕行",

    # 障碍物
    "前方有障碍物",
    "左侧有障碍物",
    "右侧有障碍物",

    # 鞋型
    "请拍摄今天出门穿的鞋子照片",
    "正在分析您的鞋子",
    "已识别鞋子，已相应调整水坑警告",
    "检测到凉鞋，将对非常浅的水坑发出警告",
    "检测到低帮鞋，将对浅水坑发出警告",
    "检测到踝靴，中等防水保护",
    "检测到靴子，良好的防水保护",
    "检测到雨靴，高度防水保护",
    "无法识别鞋子，使用默认设置",

    # 语音唤醒
    "正在监听唤醒词。说拜拜雨坑开始避障",
    "检测到唤醒词。正在启动避障",
    "收到语音指令。停止避障",
    "收到语音指令。开始避障",

    # 红绿灯
    "红灯，请停止通行",
    "黄灯，请注意",
    "绿灯，可以通行",

    # 到达 / 完成
    "已到达目的地。正在退出避障系统",
    "本次避障已完成",

    # App 介绍（较长，预生成收益最大）
    "欢迎使用不积洼步避障助手。本软件旨在帮助您在雨天出行时避开水坑与障碍物。"
    "您可以先录入您的鞋型，以便系统为您量身定制避障方案。"
    "您也可以直接对手机说：下雨了，我要出门，来唤醒导航助手。祝您出行平安。",

    # UI 引导
    "欢迎来到主界面",
    "欢迎来到声音设置界面",
    "欢迎来到拍摄录入界面，请拍摄您的鞋子照片。屏幕中心是大按钮：拍摄照片。屏幕底部是退出按钮。",
    "建议通过选择录入方式录入鞋型",
    "正在拍摄照片",
    "正在分析您的鞋子",
]


def phrase_to_filename(voice: str, text: str) -> str:
    """将 (音色, 文本) 映射到唯一文件名（md5 哈希）。"""
    key = f"{voice}:{text}"
    return hashlib.md5(key.encode("utf-8")).hexdigest() + ".mp3"


def load_index(cache_dir: Path) -> dict:
    """读取索引文件 index.json -> {md5_filename: text}。"""
    import json
    index_path = cache_dir / "index.json"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_index(cache_dir: Path, index: dict):
    """保存索引文件。"""
    import json
    index_path = cache_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


async def synthesize_one(text: str, voice: str, out_path: Path) -> bool:
    """合成单条语音，保存到 out_path。"""
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(out_path))
        return True
    except Exception as e:
        print(f"  [ERROR] 合成失败: {e}")
        return False


async def prebuild(voice: str, force: bool):
    """主预生成逻辑。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index(CACHE_DIR)   # filename -> {"text": ..., "voice": ...}

    # 反向查找：{voice:text -> filename}
    existing = {
        f"{v['voice']}:{v['text']}": fname
        for fname, v in index.items()
        if isinstance(v, dict)
    }

    total = len(FIXED_PHRASES)
    generated = 0
    skipped = 0
    failed = 0

    print(f"音色: {voice}")
    print(f"缓存目录: {CACHE_DIR}")
    print(f"共 {total} 条短语\n")

    for i, text in enumerate(FIXED_PHRASES, 1):
        filename = phrase_to_filename(voice, text)
        out_path = CACHE_DIR / filename
        key = f"{voice}:{text}"

        preview = text[:40] + ("..." if len(text) > 40 else "")

        if not force and out_path.exists() and key in existing:
            print(f"[{i:>3}/{total}] 跳过（已存在）: {preview}")
            skipped += 1
            continue

        print(f"[{i:>3}/{total}] 合成: {preview}")
        ok = await synthesize_one(text, voice, out_path)

        if ok:
            index[filename] = {"text": text, "voice": voice}
            save_index(CACHE_DIR, index)   # 每合成一条就保存，中断不丢进度
            generated += 1
        else:
            failed += 1

    print(f"\n完成！生成={generated}  跳过={skipped}  失败={failed}")
    print(f"缓存目录: {CACHE_DIR}")
    print(f"下次启动应用时将自动使用这些预生成文件，无需联网合成。")


def main():
    parser = argparse.ArgumentParser(description="预生成固定语音文件")
    parser.add_argument(
        "--voice", default="zh-CN-XiaoxiaoNeural",
        help="Edge TTS 音色 ID（默认: zh-CN-XiaoxiaoNeural）"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新生成已存在的文件"
    )
    args = parser.parse_args()

    if not EDGE_TTS_AVAILABLE:
        print("[ERROR] edge-tts 未安装，请执行:")
        print("  pip install edge-tts")
        sys.exit(1)

    asyncio.run(prebuild(args.voice, args.force))


if __name__ == "__main__":
    main()
