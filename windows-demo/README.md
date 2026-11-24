# Windows 演示：从 GitHub 复原

需要 **Windows x64、Python 3.10 x64**，以及首次安装依赖时的网络。此流程不需要原电脑的项目文件、历史权重 Release 或训练数据。

## 1. 下载并进入目录

用 Git 克隆这个私有仓库（按提示登录有访问权限的 GitHub 账号），或者在仓库首页选 **Code → Download ZIP**，解压后进入 `windows-demo`。

```powershell
git clone --branch main https://github.com/FDhzy/bujiwubu.git
cd .\bujiwubu\windows-demo
```

## 2. 校验模型、安装并启动

以下命令在含有 `verify_assets.py` 的 `windows-demo` 目录中执行。`python --version` 应显示 `Python 3.10.x`；若 PATH 指向其他 Python，就把下面两处 `python` 换成 `py -3.10`。

```powershell
python --version
python verify_assets.py
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-lock.txt
.\Start-Windows-Demo.bat
```

`verify_assets.py` 应输出四行 `OK`。如果缺文件或哈希不一致，重新下载或克隆 `main`，不要用旧模型替换。也可以运行 `Setup-Windows.ps1` 代替手动校验、创建虚拟环境和安装依赖。

## 3. 可选：复原后做自检

```powershell
.\.venv\Scripts\python.exe smoke_desktop.py
```

看到 `WINDOW_AND_INFERENCE_OK` 说明界面和合成帧推理通过。

- `Setup-Windows.ps1` 在此目录创建项目专用虚拟环境，安装列出的依赖并校验四份模型。
- `Start-Shoe-Photo-Demo.bat` 打开独立的鞋型照片建议工具；不保存照片，也不修改主界面的鞋型。
- `.\.venv\Scripts\python.exe smoke_desktop.py` 用合成帧验证界面、模型加载、积水分割、障碍检测和鞋型模型，无需摄像头；真实摄像头需由用户现场验证。
- `model-manifest.json` 保存模型大小和 SHA-256。`python verify_assets.py` 只做文件完整性检查。
- 安装依赖若报缺少 Visual C++ 运行库，请安装微软官方的 x64 Visual C++ Redistributable 后重试。

此为演示原型，不用于真实导航或判断涉水安全。原始个人照片、带临时参数的下载文件名或日志不属于运行依赖。

## 复现验收（2026-11-24）

从私有 GitHub 重新克隆初版及最终提交 `cea471f`，四份模型的大小和 SHA-256 均一致。用新建的 Python 3.10 虚拟环境安装依赖，`pip check` 通过。运行 `smoke_desktop.py` 得到 `WINDOW_AND_INFERENCE_OK`：Kivy 界面打开，真实积水 H5 与障碍、鞋型 TFLite 完成合成帧推理。合成帧上的结果不表示模型准确率。没有验证实体摄像头、真机环境或 Android APK。

依赖锁定见 [requirements-windows-lock.txt](requirements-windows-lock.txt)。
