"""
测试语音唤醒功能
"""

import threading
import time
from core.voice_activation import VoiceActivation


def on_wake():
    """唤醒回调函数"""
    print("唤醒词检测到！")


def on_command(command):
    """命令回调函数"""
    print(f"检测到命令: {command}")


def main():
    print("开始测试语音唤醒功能...")
    print("请说出唤醒词: '下雨了，我要出门'")
    print("或者尝试命令如: '停止', '开始', '我已到达目的地'")
    print("按 Ctrl+C 退出测试")
    
    # 创建语音激活实例
    voice_activation = VoiceActivation(
        on_wake=on_wake,
        on_command=on_command,
        language="zh-CN"
    )
    
    # 检查是否可用
    if voice_activation.is_available():
        print("语音激活模块可用，开始监听...")
        voice_activation.start_listening()
        
        try:
            # 保持主线程运行
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n停止测试...")
            voice_activation.stop_listening()
            voice_activation.release()
    else:
        print("语音激活模块不可用，请检查麦克风和speech_recognition库")
        if not voice_activation._microphone:
            print("- 未检测到麦克风")
        if not voice_activation._recognizer:
            print("- 未初始化识别器")


if __name__ == "__main__":
    main()