# 本地项目归档说明

- 当前可运行版本：[Windows 演示](../windows-demo/README.md)。运行不依赖本目录的历史工具或原始训练图。
- 历史代码：仓库保留比赛终版候选和曾临时回退的源码分支，见[仓库首页](../README.md)。回退分支按历史样貌留档，不承诺可运行。
- `utility-scripts/` 保留比赛目录散落的工具脚本。部分脚本依赖当时的文件布局或地图服务；它们不是演示的启动入口。
- 已核对候选目录根层 Python 工具脚本，主线已有相同内容，不重复存放。
- `water-v3/` 保存未接入演示的积水 V3 实验脚本、参数、摘要。H5 底层权重已保存在 `windows-demo/training_checkpoints/`，如需重跑实验，还需有原评测图片和标签。不要将实验评分当成现场性能保证。
- 未归档：原始训练图片包、带微信临时下载参数的文件、来源或授权未明确的图片、与参赛项目无关的目录、激活信息、编辑器缓存、虚拟环境和运行日志。它们不属于演示依赖。源盘原件未删除；需要今后再训练时，不要将 GitHub 当作上述数据的完整备份。

## 历史模型检查点

从 [私有 Release](https://github.com/FDhzy/bujiwubu/releases/tag/archive-training-checkpoints-2026-11-24) 下载四个 `training-checkpoints-*.zip` 和 `checkpoints-manifest.json`。GitHub 返回的文件大小和 SHA-256 都已经与清单比对。下载后可用 PowerShell 的 `Get-FileHash -Algorithm SHA256 training-checkpoints-*.zip` 再验一次。分别解压至同一个空文件夹，就能还原历史 `training_checkpoints/` 树。此归档不含未完成的下载文件或原始训练图片。不要覆盖 `windows-demo` 中已验证的运行权重。
