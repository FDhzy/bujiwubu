"""
修复 URL 格式
"""
import requests
import webbrowser
import time
import urllib.parse

AMAP_WEB_KEY = "16baecbb1c842b94eb8c0ce06e5955b8"

# 搜索
keyword = input("请输入目的地：")
print(f"\n搜索：{keyword}")

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
    print(f"\n找到 {len(pois)} 个结果:\n")
    
    for i, poi in enumerate(pois):
        name = poi.get("name", "")
        address = poi.get("address", "")
        location = poi.get("location", "")
        print(f"[{i}] {name}")
        print(f"    地址：{address}")
        print(f"    坐标：{location}\n")
    
    # 选择第一个
    if pois:
        poi = pois[0]
        name = poi.get("name", "")
        location = poi.get("location", "")
        
        # 对地名进行 URL 编码
        encoded_name = urllib.parse.quote(name)
        
        if location:
            lon, lat = location.split(",")
            # 方式 1：使用 query 参数
            map_url = f"https://www.amap.com/search?query={encoded_name}"
            # 方式 2：直接在 hash 中传入坐标
            map_url2 = f"https://www.amap.com/?search={encoded_name}&geoobj={lon}|{lat}"
            # 方式 3：使用最简单的形式
            map_url3 = f"https://www.amap.com/search?query={encoded_name}&mapzoom=15&center={lon},{lat}"
        else:
            map_url = f"https://www.amap.com/search?query={encoded_name}"
            map_url2 = map_url
            map_url3 = map_url
        
        print(f"\n尝试以下 URL:")
        print(f"URL1: {map_url}")
        print(f"URL2: {map_url2}")
        print(f"URL3: {map_url3}")
        
        # 先测试 URL1
        print(f"\n调用 webbrowser.open()...")
        result = webbrowser.open(map_url)
        print(f"返回值：{result}")
        
        print(f"\n等待 8 秒...")
        time.sleep(8)
        
else:
    print("\n搜索失败或无结果")
    print(f"返回数据：{data}")

print("\n程序结束")