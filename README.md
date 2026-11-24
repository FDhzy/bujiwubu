# 不积洼步

魔搭 AI Hackathon 的低视力出行辅助原型，探索用摄像头识别障碍物、积水和鞋型。**当前仅用于演示与实验，不能用于实际导航或判断道路安全。**

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
