"""
修改 navigation_input_screen.py - 使用新的 NavigationHelper
"""
import os

base = r"d:\-- 不积洼步_10日改 (1)"

# 找到 navigation_input_screen.py
for root, dirs, files in os.walk(base):
    if 'navigation_input_screen.py' in files:
        py_path = os.path.join(root, 'navigation_input_screen.py')
        print(f"找到文件：{py_path}")
        
        with open(py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. 添加 NavigationHelper 导入
        if 'from .native_input_dialog import NativeInputDialog' in content:
            # 替换原有导入
            content = content.replace(
                'from .native_input_dialog import NativeInputDialog',
                '# from .native_input_dialog import NativeInputDialog\nfrom .navigation_helper import NavigationHelper'
            )
        elif 'NavigationHelper' not in content:
            # 在其他导入后面添加
            import_section_end = content.find('\n\nclass ')
            if import_section_end > 0:
                # 找到最后一个 import 语句
                last_import = content.rfind('\nimport ', 0, import_section_end)
                if last_import > 0:
                    next_line = content.find('\n', last_import)
                    insert_pos = next_line + 1
                    content = content[:insert_pos] + '\nfrom ui.navigation_helper import NavigationHelper\n' + content[insert_pos:]
        
        # 2. 修改 _show_input_popup 方法
        old_method = '''def _show_input_popup(self):
        """Show destination input popup."""
        try:
            from .native_input_dialog import NativeInputDialog
            dialog = NativeInputDialog(app=self.app)
            dialog.show()
        except Exception as e:
            self.logger.error(f"Failed to show input dialog: {e}")'''
        
        new_method = '''def _show_input_popup(self):
        """Show destination input using native Windows dialog."""
        try:
            # Use simple Tkinter dialog for Chinese IME support
            import threading
            import tkinter as tk
            from tkinter import simpledialog
            
            def input_thread():
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                
                destination = simpledialog.askstring(
                    "导航目的地输入",
                    "请输入目的地名称（如：天安门）",
                    parent=root
                )
                
                if destination and destination.strip():
                    dest_text = destination.strip()
                    # Search and navigate
                    self._search_and_navigate(dest_text)
                
                root.destroy()
            
            threading.Thread(target=input_thread, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"Failed to show input dialog: {e}")'''
        
        if old_method in content:
            content = content.replace(old_method, new_method)
            print("已修改 _show_input_popup 方法")
        
        # 3. 添加搜索和导航方法
        search_method = '''
    def _search_and_navigate(self, keyword):
        """Search destination and open Amap web version."""
        try:
            self.logger.info(f"Searching for: {keyword}")
            
            # Create navigation helper
            nav_helper = NavigationHelper(app=self.app)
            
            def on_search_complete(results):
                if results:
                    self.logger.info(f"Found {len(results)} results")
                    # Use first result
                    nav_helper.open_amap_web(results[0])
                    
                    # Start camera navigation
                    Clock.schedule_once(lambda dt: self._start_camera_nav(results[0]), 2.0)
                else:
                    self.logger.warning("No results found")
                    if self.app and hasattr(self.app, 'feedback_controller'):
                        self.app.feedback_controller.speak_text("未找到相关地点")
            
            nav_helper.search_destination(keyword, on_search_complete)
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            if self.app and hasattr(self.app, 'feedback_controller'):
                self.app.feedback_controller.speak_text("搜索失败")
    
    def _start_camera_nav(self, poi):
        """Switch back to main screen and start obstacle-avoidance camera."""
        if self.app is None:
            return
        
        # Store mode and destination in app for reference
        self.app.navigation_mode = 'navigation'
        self.app.navigation_destination = poi.get('name', '')
        
        # Switch to main screen
        self.app.screen_manager.current = 'main'
        
        # Trigger camera start
        Clock.schedule_once(lambda dt: self._trigger_nav_start(), 0.2)
    
    def _trigger_nav_start(self):
        """Actually start obstacle detection on main screen."""
        if self.app and hasattr(self.app, 'main_screen'):
            self.app.main_screen._is_navigating = True
            self.app.start_navigation()
            dest = getattr(self.app, 'navigation_destination', '')
            self.app.main_screen._set_status(f"导航中 → {dest}")
            self.app.main_screen._update_button_state()
'''
        
        # 在 _start_camera_nav 旧方法的位置插入新方法
        if 'def _start_camera_nav(self, poi: dict):' in content:
            # 替换旧方法
            old_start_method = '''    def _start_camera_nav(self, poi: dict):
        """Switch back to main screen and start obstacle-avoidance camera."""
        if self.app is None:
            return

        # Store mode and destination in app for reference
        self.app.navigation_mode = 'navigation'
        self.app.navigation_destination = poi.get('name', '')

        # Switch to main screen
        self.app.screen_manager.current = 'main'

        # Trigger camera start
        Clock.schedule_once(lambda dt: self._trigger_nav_start(), 0.2)

    def _trigger_nav_start(self):
        """Actually start obstacle detection on main screen."""
        if self.app and hasattr(self.app, 'main_screen'):
            self.app.main_screen._is_navigating = True
            self.app.start_navigation()
            dest = getattr(self.app, 'navigation_destination', '')
            self.app.main_screen._set_status(f"导航中 → {dest}")
            self.app.main_screen._update_button_state()'''
            
            if old_start_method in content:
                content = content.replace(old_start_method, search_method)
                print("已添加搜索和导航方法")
        
        # 保存修改后的文件
        with open(py_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 已保存：{py_path}")
        
        break
else:
    print("未找到 navigation_input_screen.py 文件")

print("\n完成！")