import os

# 找到 navigation_input_screen.kv 文件
base = r"d:\-- 不积洼步_10日改 (1)"

for root, dirs, files in os.walk(base):
    if 'navigation_input_screen.kv' in files:
        kv_path = os.path.join(root, 'navigation_input_screen.kv')
        print(f"找到 KV 文件：{kv_path}")
        
        with open(kv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换 Label 为 TextInput
        old_code = '''    # Static placeholder label (not editable)
    Label:
        text: "请点击下方搜索按钮，在弹出的窗口中输入目的地"
        font_name: 'ChineseFont'
        font_size: sp(20)
        color: 0.5, 0.5, 0.5, 1
        halign: 'center'
        valign: 'middle'
        text_size: self.size'''
        
        new_code = '''    # 目的地输入框
    NavTextInput:
        id: destination_input
        hint_text: '请输入目的地（如：天安门）'
        font_name: 'ChineseFont'
        font_size: sp(20)
        multiline: False
        background_color: 1, 1, 1, 0.95
        foreground_color: 0.07, 0.16, 0.18, 1
        cursor_color: 0.18, 0.49, 0.93, 1
        padding: [dp(15), dp(10), dp(15), dp(10)]
        on_text_validate: root.on_search_destination()'''
        
        if old_code in content:
            content = content.replace(old_code, new_code)
            with open(kv_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("已修改 KV 文件，将 Label 改为 TextInput")
        else:
            print("未找到要替换的代码段")
        
        break

print("\n完成！")