# 不积洼步（Blind Navigation Assistant）

> 本项目为 **2026 年初参赛作品**，参加魔搭社区（ModelScope）与 Datawhale 发起的 **AI Hackathon Tour 2026 高校联赛**，赛道 3「小有可为」公益应用。该届全国总决赛于 **2026 年 3 月 21 日**在南京举行，见[主办方赛事记录](https://www.modelscope.cn/events/179)。

## 一、项目概述

不积洼步是面向视障人群的出行辅助原型，重点识别路面水坑与积水，并兼顾行人、车辆等立体障碍物。系统以摄像头图像为输入，在同一画面中标出疑似积水区域和障碍物，便于查看与调试检测结果。

## 二、当前实施方案

| 模块 | 实现 |
| --- | --- |
| 图像采集与推理 | Python + Kivy 界面接入摄像头，推理引擎并行处理积水分割和障碍物检测，汇总后叠加到预览画面。 |
| 积水分割 | Windows 演示使用本地 Keras H5 模型生成像素级分割结果，经阈值过滤、形态学去噪和小连通区域过滤后，在界面标记疑似积水。 |
| 障碍物检测 | 使用轻量 YOLO 系列 TFLite 模型输出检测框，经置信度过滤和非极大值抑制（NMS）去重后显示。 |
| 鞋型设置 | Windows 主界面由用户手动选择鞋型。独立照片工具可以给出靴类建议；证据不足时提示手选，不会自动写入设置。 |
| 交互与反馈 | 源码包含中英语音、振动、语音唤醒和方向提示模块。Windows 演示以画面标注为主，不开启语音唤醒、不输出行走指令。 |

**运行平台**：当前提供本地 Windows 演示，已验证界面启动、模型加载和样本推理。Android 源码包含 Buildozer / python-for-android 配置（目标 API 33、最低 API 26，配置了 arm64-v8a 与 armeabi-v7a），未产出经验证的 APK。

**使用边界**：检测结果仅用于演示与测试，不能用于实际导航、判断可通行路线或估计安全涉水深度。

## 从哪里开始

| 我想做什么 | 去哪里 | 这个版本的状态 |
| --- | --- | --- |
| 运行 Windows 演示 | [安装与复现说明](windows-demo/README.md) | 包含启动器、完整运行模型和独立鞋型入口 |
| 看主线源码 | [`main`](https://github.com/FDhzy/bujiwubu/tree/main) | 保留原有项目代码与 Android 构建配置；不是最新 Windows 演示包 |
| 找比赛终版候选、恢复模型 | [终版候选归档](https://github.com/FDhzy/bujiwubu/tree/archive/final-candidate-2026-11-23) | 固定归档；含模型分片、校验清单和恢复脚本 |
| 看 Windows 鞋型实验 | [鞋型照片建议与对照](https://github.com/FDhzy/bujiwubu/blob/improve/windows-models-2026-11-23/desktop-model-evaluation/README-Model-V2.md) | 仅给照片建议，不自动设置鞋型 |
| 看积水 V3 权重和评估 | [积水 V3 实验目录](https://github.com/FDhzy/bujiwubu/tree/improve/windows-models-2026-11-23/desktop-model-evaluation/water-v3) | 实验候选，未接入当前演示 |
| 对照赛时回退源码 | [回退源码归档](https://github.com/FDhzy/bujiwubu/tree/archive/rollback-source-2026-11-23) | 历史快照，有已知语法问题，不是可运行发布版 |
| 查 Android 打包验证 | [独立验证分支](https://github.com/FDhzy/bujiwubu/tree/verify/final-candidate-2026-11-23) · [构建记录](https://github.com/FDhzy/bujiwubu/actions/runs/45183317656) | 未生成经验证的 APK；本轮不继续打包 |

## 怎么运行或复现

**在 Windows 上运行：**克隆本仓库后，进入 [windows-demo](windows-demo/README.md)，安装 Python 3.10 x64 和项目依赖，再启动演示。不依赖旧电脑桌面文件。

**恢复终版候选的归档模型：**切到终版候选归档，在仓库根目录用 Python 3.10+ 执行 `python scripts/restore_models.py`。脚本按[模型清单](https://github.com/FDhzy/bujiwubu/blob/archive/final-candidate-2026-11-23/model-archive/manifest.json)恢复并校验文件。这里恢复的是比赛候选的模型，不是 Windows 积水 V3 所需的 H5 底层权重。V3 在 GitHub 只保存新分割层及其底层权重校验值；复现需要原本地 H5 和评测数据。

## 主线代码导航

- [`main.py`](main.py)：应用入口。
- [`core/`](core/) · [`models/`](models/) · [`processing/`](processing/)：相机、检测模型和结果处理。
- [`ui/`](ui/) · [`guidance/`](guidance/)：界面与提示逻辑。
- [`buildozer.spec`](buildozer.spec)：Android 构建配置。

[归档索引](project-archive/README.md)列出历史工具、实验以及未纳入仓库的资料。Windows 演示已从 GitHub 重新克隆并通过合成帧检查；未经现场摄像头或安卓真机验证。
