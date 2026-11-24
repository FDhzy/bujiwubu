"""
修复高德地图导航 - 使用网页版而不是 amapuri 协议
"""
import os

base = r"d:\-- 不积洼步_10日改 (1)"

# 找到所有 Python 文件并检查是否包含导航相关代码
for root, dirs, files in os.walk(base):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否包含导航相关代码
                if 'amap' in content.lower() or '高德' in content or 'webbrowser' in content:
                    print(f"\n找到相关文件：{filepath}")
                    
                    # 查找并替换 amapuri 协议调用
                    if 'amapuri://' in content:
                        print("  -> 发现 amapuri 协议")
                        content = content.replace(
                            'amapuri://route/plan/?dlat={lat}&dlon={lon}&dname={name}&dev=0&t=0',
                            'https://www.amap.com/search?query={name}'
                        )
                    
                    # 查找 webbrowser.open 调用
                    if 'webbrowser.open(' in content:
                        print("  -> 发现 webbrowser.open 调用")
                        # 显示相关代码行
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'webbrowser.open' in line:
                                print(f"     行{i+1}: {line.strip()}")
                    
                    # 保存修改后的内容
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print("  ✓ 已处理")
                    
            except Exception as e:
                pass

print("\n\n处理完成！")