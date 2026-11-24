import os
import sys

# 找到主程序目录
base = r"d:\-- 不积洼步_10日改 (1)"

# 搜索所有 Python 文件
found_files = []
for root, dirs, files in os.walk(base):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否包含任何地图相关代码
                if any(keyword in content for keyword in ['amap', '高德', 'webbrowser', 'uri', 'route']):
                    found_files.append(filepath)
                    print(f"\n{filepath}")
                    
                    # 显示相关的行
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if any(kw in line.lower() for kw in ['amap', '高德', 'webbrowser', 'uri', 'route', 'open(']):
                            if not line.strip().startswith('#'):
                                print(f"  {i}: {line.strip()}")
            
            except Exception as e:
                pass

print(f"\n\n共找到 {len(found_files)} 个相关文件")
print("\n请检查上面列出的文件，看看哪个包含了调起高德地图的代码")