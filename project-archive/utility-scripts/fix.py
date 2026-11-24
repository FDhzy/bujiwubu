import os

base = r"d:\-- 不积洼步_10日改 (1)"

for root, dirs, files in os.walk(base):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'amapuri://' in content:
                    print(f"找到：{filepath}")
                    new_content = content.replace(
                        'amapuri://route/plan/?dlat={lat}&dlon={lon}&dname={name}&dev=0&t=0',
                        'https://www.amap.com/search?query={name}'
                    )
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"已修复：{filepath}")
            except Exception as e:
                print(f"处理 {filepath} 时出错：{e}")

print("\n完成！请重启你的应用程序。")