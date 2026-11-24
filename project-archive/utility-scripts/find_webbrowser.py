import os

base = r"d:\-- 不积洼步_10日改 (1)"

print("查找包含 webbrowser 的文件：")
for root, dirs, files in os.walk(base):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'webbrowser' in content or '_open_amap' in content:
                    print(f"\n{filepath}")
                    # 显示相关行
                    for i, line in enumerate(content.split('\n'), 1):
                        if 'webbrowser' in line or '_open_amap' in line or 'open(' in line and 'http' in line:
                            print(f"  行{i}: {line.strip()}")
            except Exception as e:
                pass

print("\n\n查找完成")