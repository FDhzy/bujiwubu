import os

base = r"d:\-- 不积洼步_10日改 (1)"

print("正在搜索包含 amapuri 的文件...\n")

found_count = 0

for root, dirs, files in os.walk(base):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'amapuri://' in content:
                    found_count += 1
                    print(f"\n[{found_count}] {filepath}")
                    
                    # 显示包含 amapuri 的行
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if 'amapuri' in line:
                            print(f"    行{i}: {line.strip()}")
                    
                    # 询问是否修改
                    answer = input("\n要修改这个文件吗？(y/n): ")
                    if answer.lower() == 'y':
                        # 替换 amapuri 协议
                        new_content = content.replace(
                            'amapuri://route/plan/?dlat={lat}&dlon={lon}&dname={name}&dev=0&t=0',
                            'https://www.amap.com/search?query={name}'
                        )
                        
                        # 也要处理多行的情况
                        old_multi = '''f"amapuri://route/plan/"
                    f"?dlat={lat}&dlon={lon}&dname={name}"
                    f"&dev=0&t=0"'''
                        
                        new_multi = '''f"https://www.amap.com/search?query={name}&geoobj={lon}|{lat}"'''
                        
                        new_content = new_content.replace(old_multi, new_multi)
                        
                        # 保存
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        
                        print(f"✓ 已修改并保存")
            
            except Exception as e:
                print(f"处理 {filepath} 时出错：{e}")

print(f"\n\n总共找到 {found_count} 个包含 amapuri 的文件")
print("修复完成！")