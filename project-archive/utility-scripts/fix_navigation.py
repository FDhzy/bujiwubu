import os

base = r"d:\-- 不积洼步_10日改 (1)"

# 找到 navigation_input_screen.py
for root, dirs, files in os.walk(base):
    if 'navigation_input_screen.py' in files:
        py_path = os.path.join(root, 'navigation_input_screen.py')
        print(f"找到文件：{py_path}")
        
        with open(py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否有 _show_input_popup 方法
        if '_show_input_popup' in content or 'DestinationInputPopup' in content or 'NativeInputDialog' in content:
            print("发现输入对话框相关代码")
            
            # 移除对 NativeInputDialog 或 DestinationInputPopup 的引用
            # 改为直接显示一个 TextInput
            old_import = '''from .native_input_dialog import NativeInputDialog'''
            new_import = '''# from .native_input_dialog import NativeInputDialog'''
            
            content = content.replace(old_import, new_import)
            
            # 修改 _show_input_popup 方法
            old_method = '''def _show_input_popup(self):
        """Show destination input popup."""
        try:
            from .native_input_dialog import NativeInputDialog
            dialog = NativeInputDialog(app=self.app)
            dialog.show()
        except Exception as e:
            self.logger.error(f"Failed to show input dialog: {e}")'''
            
            new_method = '''def _show_input_popup(self):
        """Show destination input - focus on the text input field."""
        try:
            # Get the input field and set focus
            input_field = self.ids.get('destination_input')
            if input_field:
                input_field.focus = True
                self.logger.info("Input field focused")
            else:
                self.logger.warning("Input field not found")
        except Exception as e:
            self.logger.error(f"Failed to focus input field: {e}")'''
            
            content = content.replace(old_method, new_method)
            
            with open(py_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("已修改 Python 文件")
        else:
            print("未找到输入对话框相关代码")
        
        break

print("\n完成！")