# 不积洼步（Blind Navigation Assistant）

> **参赛项目**：魔搭 AI Hackathon Tour 高校联赛 · 赛道 3「小有可为」公益应用。[赛道介绍](https://njsj.nanjing.gov.cn/xwdt/202603/t20260325_5812173.html)

## 一、项目概述

不积洼步是我们此前为比赛开发的视障出行辅助原型，聚焦一个容易被忽略的问题：**避开路面积水**。常规避障往往关注行人、车辆等立体障碍；水坑贴着地面，反光又容易与干地混淆，值得单独识别。

参赛方案以手机摄像头为输入，并行检测积水与障碍物，再用语音和振动提示。我们想探索的不只是告诉用户“前方有风险”，还包括在条件允许时提示避让方向，并根据鞋型调整提醒策略、让用户自定义播报。绕行方向、步数换算和鞋型对应的水深阈值都是原型中的设计规则，**不等于已经测准距离、水深或验证了路线安全**。

## 二、原型架构与当前实现

| 模块 | 参赛方案 | 当前验证状态 |
| --- | --- | --- |
| 推理引擎 | 并行处理积水、障碍物和传统视觉反射特征，汇总检测结果 | Windows 演示验证了模型加载和样本推理；不把反光规则直接并入积水掩膜，也未验证手机实时帧率 |
| 积水检测 | 像素级分割，配合阈值、形态学去噪和反光分析 | 修正了 Windows 模型输入输出处理。V3 在留出样本上将有误报的干地图从 18/18 降到 2/18，但两批积水图的完全漏检从 0/70 升到 8/70，因此只归档、未启用 |
| 障碍物检测 | YOLO 系列轻量模型、检测框过滤和去重 | Windows 调整了 TFLite 输出解析及置信度；只在可用的带标注类别上做了离线检测，不宣称覆盖全部 COCO 类别 |
| 鞋型适配 | 支持手选与照片识别，按鞋型配置提醒 | 主演示仍手选；独立照片实验在同一批 180 张商品图上，强制分类由 104/180 提高到 126/180。不确定时不建议，不自动改鞋型 |
| 引导与交互 | 探索方向或步数提示、中英语音、振动、播报设置和语音唤醒 | Windows 演示不输出行走指令，也不开启语音唤醒；原型方案保留在代码中，未验证真实道路表现 |

**平台与模型**：Python + Kivy；Android 打包配置使用 Buildozer 和 python-for-android，目标 API 33、最低 API 26，配置了 arm64-v8a、armeabi-v7a。配置声明了摄像头、录音、振动、唤醒锁和网络权限；这些是构建配置，不代表 APK 已验证。终版候选归档保存了 `water_model.tflite`（积水）、`obstacle_best.tflite`（障碍物）、`shoe_model.tflite`（鞋型）；Windows 演示的积水基线使用本地 H5。

**使用边界**：这是参赛原型和后续离线实验，尚未验证真实道路上的安全性，不能用来实际导航或判断能否涉水。

## 从哪里开始

| 我想做什么 | 去哪里 | 这个版本的状态 |
| --- | --- | --- |
| 看主线源码 | [`main`](https://github.com/FDhzy/bujiwubu/tree/main) | 保留原有项目代码与 Android 构建配置；不是最新 Windows 演示包 |
| 找比赛终版候选、恢复模型 | [终版候选归档](https://github.com/FDhzy/bujiwubu/tree/archive/final-candidate-2026-11-23) | 固定归档；含模型分片、校验清单和恢复脚本 |
| 看 Windows 鞋型实验 | [鞋型照片建议与对照](https://github.com/FDhzy/bujiwubu/blob/improve/windows-models-2026-11-23/desktop-model-evaluation/README-Model-V2.md) | 仅给照片建议，不自动设置鞋型 |
| 看积水 V3 权重和评估 | [积水 V3 实验目录](https://github.com/FDhzy/bujiwubu/tree/improve/windows-models-2026-11-23/desktop-model-evaluation/water-v3) | 干地误报变少，但漏检增多，**未启用** |
| 对照赛时回退源码 | [回退源码归档](https://github.com/FDhzy/bujiwubu/tree/archive/rollback-source-2026-11-23) | 历史快照，有已知语法问题，不是可运行发布版 |
| 查 Android 打包验证 | [独立验证分支](https://github.com/FDhzy/bujiwubu/tree/verify/final-candidate-2026-11-23) · [构建记录](https://github.com/FDhzy/bujiwubu/actions/runs/45183317656) | 未生成经验证的 APK；本轮不继续打包 |

## 怎么运行或复现

**在原 Windows 电脑上演示：**进入桌面的 `bujiwubu-windows-improved`，双击 `Start-Windows-Demo.bat`。保留同级的 `bujiwubu-windows-demo` 文件夹，启动脚本会从那里加载依赖。Windows 完整运行副本没有同步到 GitHub；GitHub 实验分支不能直接替代它。鞋型独立试用入口在桌面的 `bujiwubu-windows-model-v2/Start-Shoe-Photo-Demo.bat`。

**恢复终版候选的归档模型：**切到终版候选归档，在仓库根目录用 Python 3.10+ 执行 `python scripts/restore_models.py`。脚本按[模型清单](https://github.com/FDhzy/bujiwubu/blob/archive/final-candidate-2026-11-23/model-archive/manifest.json)恢复并校验文件。这里恢复的是比赛候选的模型，不是 Windows 积水 V3 所需的 H5 底层权重。V3 在 GitHub 只保存新分割层及其底层权重校验值；复现需要原本地 H5 和评测数据。

## 主线代码导航

- [`main.py`](main.py)：应用入口。
- [`core/`](core/) · [`models/`](models/) · [`processing/`](processing/)：相机、检测模型和结果处理。
- [`ui/`](ui/) · [`guidance/`](guidance/)：界面与提示逻辑。
- [`buildozer.spec`](buildozer.spec)：Android 构建配置。

后续实验优先放到实验分支，记录前后指标和是否启用。历史归档只用于恢复与对照。照片、训练数据、个人配置和运行日志不随实验提交。
