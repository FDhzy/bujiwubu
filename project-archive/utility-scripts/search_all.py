import os

base = r"d:\-- 不积洼步_10日改 (1)"

print("搜索整个项目中的 amapuri...\n")

count = 0
for root, dirs, files in os.walk(base):
    for file in files:
        filepath = os.path.join(root, file)
        try:
            # 尝试读取各种文件
            if file.endswith(('.py', '.kv', '.txt', '.md', '.json', '.yaml', '.yml')):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if 'amapuri' in content:
                    count += 1
                    print(f"[{count}] {filepath}")
                    
                    # 显示相关行
                    for i, line in enumerate(content.split('\n')[:200], 1):
                        if 'amapuri' in line:
                            print(f"    行{i}: {line.strip()}")
        except:
            pass

print(f"\n\n总共找到 {count} 个文件包含 amapuri")