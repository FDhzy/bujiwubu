"""
导航功能模块 - 使用网页版高德地图
完全避开 amapuri 协议问题
"""
import os
import requests
import webbrowser
import urllib.parse
from kivy.clock import Clock
from kivy.logger import Logger

AMAP_WEB_KEY = "16baecbb1c842b94eb8c0ce06e5955b8"


class NavigationHelper:
    """导航助手 - 搜索目的地并打开高德地图网页版"""
    
    def __init__(self, app=None):
        self.app = app
        self.logger = Logger
    
    def search_destination(self, keyword, callback=None):
        """
        搜索目的地
        
        Args:
            keyword: 目的地关键词
            callback: 回调函数，接收搜索结果列表
        """
        try:
            url = "https://restapi.amap.com/v3/place/text"
            params = {
                "keywords": keyword,
                "key": AMAP_WEB_KEY,
                "city": "全国",
                "offset": 10,
                "page": 1,
                "extensions": "base"
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get("status") == "1" and int(data.get("count", 0)) > 0:
                pois = data.get("pois", [])
                results = []
                for poi in pois:
                    results.append({
                        "name": poi.get("name", ""),
                        "address": poi.get("address", ""),
                        "location": poi.get("location", "")  # 经度，纬度
                    })
                
                self.logger.info(f"找到 {len(results)} 个搜索结果")
                
                if callback:
                    Clock.schedule_once(lambda dt: callback(results))
                return results
            else:
                self.logger.warning("高德 API 返回无结果")
                if callback:
                    Clock.schedule_once(lambda dt: callback([]))
                return []
        
        except Exception as e:
            self.logger.error(f"搜索失败：{e}")
            if callback:
                Clock.schedule_once(lambda dt: callback([]))
            return []
    
    def open_amap_web(self, poi):
        """
        在浏览器中打开高德地图网页版
        
        Args:
            poi: 地点信息字典，包含 name, location 等字段
        """
        try:
            name = poi.get("name", "目的地")
            location = poi.get("location", "")
            
            # 对地名进行 URL 编码
            encoded_name = urllib.parse.quote(name)
            
            if location:
                lon, lat = location.split(",")
                # 使用带坐标的 URL 格式
                url = f"https://www.amap.com/search?query={encoded_name}&mapzoom=15&center={lon},{lat}"
            else:
                # 没有坐标，只搜索名称
                url = f"https://www.amap.com/search?query={encoded_name}"
            
            self.logger.info(f"打开高德地图网页：{url}")
            
            # 打开浏览器
            result = webbrowser.open(url)
            
            if result:
                self.logger.info("浏览器已打开")
                # 语音反馈
                if self.app and hasattr(self.app, 'feedback_controller'):
                    Clock.schedule_once(
                        lambda dt: self.app.feedback_controller.speak_text(
                            f"已在浏览器中打开高德地图，导航至{name}"
                        )
                    )
            else:
                self.logger.warning("浏览器打开失败")
            
            return result
        
        except Exception as e:
            self.logger.error(f"打开地图失败：{e}")
            return False


# 测试代码
if __name__ == "__main__":
    def on_search_complete(results):
        if results:
            print(f"找到 {len(results)} 个结果:")
            for i, r in enumerate(results):
                print(f"  [{i}] {r['name']}: {r['address']}")
            
            # 打开第一个结果
            if results:
                print(f"\n准备打开：{results[0]['name']}")
                helper.open_amap_web(results[0])
        else:
            print("未找到结果")
    
    helper = NavigationHelper()
    
    # 测试搜索
    keyword = input("请输入目的地：")
    helper.search_destination(keyword, on_search_complete)
    
    # 保持程序运行一会儿
    import time
    time.sleep(5)