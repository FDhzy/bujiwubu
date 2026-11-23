# 不积洼步｜终版候选归档

魔搭 AI Hackathon 的低视力出行辅助原型：根据摄像头识别障碍物、水坑和鞋型，通过语音或震动给出提醒。它是比赛原型，不能当作独立的出行安全工具。

## 选的是什么版本

- 以桌面副本 `bujiwubu` 为终版候选。忽略 Windows 及 Git 的换行差异后，入口 `main.py` 和主要界面、推理代码与仓库 `main` 一致。
- 没把桌面版的两个差异直接覆盖上来：保留仓库已有的 `core/types.py` 的 `import time` 和 `buildozer.spec` 的 Android 本地编译配方设置。桌面打包配置另外列了 `tflite_runtime`，未验证其在该 Android 工具链能否构建，所以不替换现有路径。
- [比赛期间回退目录的源码](https://github.com/FDhzy/bujiwubu/tree/archive/rollback-source-2026-11-23) 另存，不混入这个分支。
- 不纳入训练集、试跑照片、训练中间权重、打包缓存、日志和个人运行设置。

## 恢复比赛候选的模型

在这个分支的仓库根目录执行（Python 3.10+，不需要额外安装 Python 包）：

```bash
python scripts/restore_models.py
```

障碍物模型和备用 DeepLabV3 文件已放在原路径。鞋型、水坑模型较大，以分片存档；脚本按顺序恢复到 `training_checkpoints/`，逐片与整文件都校验 SHA-256，不会覆盖不同的现有文件。完整清单见 [model-archive/manifest.json](model-archive/manifest.json)。

| 文件 | 内容 |
| --- | --- |
| `main.py` | Kivy 应用入口 |
| `core/`、`models/`、`processing/` | 相机、推理与检测结果 |
| `guidance/`、`ui/` | 语音、振动和界面 |
| `buildozer.spec`、`p4a_recipes/` | Android 打包配置 |

Linux 下准备 Buildozer 与 Android 工具链后，可按 [打包说明](BUILD_INSTRUCTIONS.md) 试跑 `buildozer android debug`。这个分支的 GitHub Actions 手动构建会先恢复模型；此次归档没有触发构建，也未验证 APK、手机摄像头或推理表现。
