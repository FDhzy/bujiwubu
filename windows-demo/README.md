# Windows 演示

这是可独立安装的 Windows 源码演示，不依赖原电脑桌面上的旧运行目录。需要 Windows x64、64 位 Python 3.10 和首次安装依赖时的网络。Android 未在此流程验证。

```powershell
cd windows-demo
powershell -NoProfile -ExecutionPolicy Bypass -File .\Setup-Windows.ps1
.\Start-Windows-Demo.bat
```

- `Setup-Windows.ps1` 在此目录创建项目专用虚拟环境，安装列出的依赖并校验四份模型。
- `Start-Shoe-Photo-Demo.bat` 打开独立的鞋型照片建议工具；不保存照片，也不修改主界面的鞋型。
- `.\.venv\Scripts\python.exe smoke_desktop.py` 用合成帧验证界面、模型加载、积水分割、障碍检测和鞋型模型，无需摄像头；真实摄像头需由用户现场验证。
- `model-manifest.json` 保存模型大小和 SHA-256。`python verify_assets.py` 只做文件完整性检查。
- 安装依赖若报缺少 Visual C++ 运行库，请安装微软官方的 x64 Visual C++ Redistributable 后重试。

此为演示原型，不用于真实导航或判断涉水安全。原始个人照片、带临时参数的下载文件名或日志不属于运行依赖。
