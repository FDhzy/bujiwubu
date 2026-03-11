"""
Simple TTS test to verify audio output is working
"""
import pyttsx3

print("Initializing TTS engine...")
engine = pyttsx3.init()

# Set properties
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Get available voices
voices = engine.getProperty('voices')
print(f"\nAvailable voices ({len(voices)}):")
for i, voice in enumerate(voices):
    print(f"  {i}: {voice.name}")

# Try to set Chinese voice
for voice in voices:
    if "chinese" in voice.name.lower() or "huihui" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        print(f"\nUsing voice: {voice.name}")
        break

# Test speech
test_messages = [
    "Hello, this is a test",
    "你好，这是一个测试",
    "准备就绪。请将摄像头向前"
]

print("\n" + "="*50)
print("Starting TTS test...")
print("="*50)

for i, msg in enumerate(test_messages, 1):
    print(f"\n{i}. Speaking: {msg}")
    try:
        engine.say(msg)
        engine.runAndWait()
        print(f"   ✓ Finished speaking")
    except Exception as e:
        print(f"   ✗ Error: {e}")

print("\n" + "="*50)
print("TTS test completed!")
print("="*50)
print("\nDid you hear the voice? (Y/N)")
print("If not, please check:")
print("  1. System volume is turned up")
print("  2. Speakers/headphones are connected")
print("  3. No other application is blocking audio")
