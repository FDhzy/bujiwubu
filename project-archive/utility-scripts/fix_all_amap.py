import os

# 找到所有包含 amapuri 的文件并修改
base = r"d:\-- 不积洼步_10日改 (1)"

files_to_fix = [
    r"ui\destination_input_popup.py",
    r"ui\native_input_dialog.py", 
    r"ui\navigation_input_screen.py"
]

for rel_path in files_to_fix:
    # 尝试不同的路径组合
    for root, dirs, files in os.walk(base):
        if rel_path in [os.path.join(r, f) for r, d, fs in os.walk(root) for f in fs]:
            filepath = os.path.join(root, *rel_path.split('\\'))
            if os.path.exists(filepath):
                print(f"\n处理文件：{filepath}")
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 替换 amapuri 为网页版
                old_patterns = [
                    'amapuri://route/plan/?dlat={lat}&dlon={lon}&dname={name}&dev=0&t=0',
                    '''f"amapuri://route/plan/"\n                    f"?dlat={lat}&dlon={lon}&dname={name}"\n                    f"&dev=0&t=0"''',
                    '''f"amapuri://route/plan/"\n                f"?dlat={lat}&dlon={lon}&dname={name}"\n                f"&dev=0&t=0"''',
                ]
                
                new_code = '''f"https://www.amap.com/search?query={name}&geoobj={lon}|{lat}"'''
                
                modified = False
                for old in old_patterns:
                    if old in content:
                        content = content.replace(old, new_code)
                        modified = True
                        print(f"  ✓ 已替换 amapuri 协议")
                
                # 也要替换 AMAP_APP_SCHEME
                if 'AMAP_APP_SCHEME = "amapuri"' in content:
                    content = content.replace('AMAP_APP_SCHEME = "amapuri"', 'AMAP_APP_SCHEME = "https"  # Changed to web version')
                    print(f"  ✓ 已修改 AMAP_APP_SCHEME")
                    modified = True
                
                if modified:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✓ 文件已保存")
                break
    
    else:
        print(f"未找到：{rel_path}")

print("\n\n批量修复完成！")