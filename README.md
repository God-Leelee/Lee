# 澄鉴 CLEARTRACE

图片与视频材料反欺诈原型 **1.8.0 研究预览版**  
同济大学 李洪宇 · 众安 1024 AI Coding 大赛

可在 Windows + NVIDIA GPU 本机运行的取证工作台，覆盖 AI 图片、AI 视频、传统篡改和统一报告。界面采用简洁布局与浅色半透明玻璃材质，材料保留在本机。

![本地实际工作台，仅含自制工程样例](docs/images/workbench.png)

**当前公开混合研究测试达到数值门槛，文档局部 AI 修改尚未达标。** 请结合下方范围阅读成绩。当前版本不代表全部真实理赔材料已通过正式验收。

| 已核验结果 | 本轮实测 |
|---|---|
| 100 套混合材料，70 来源真实 / 30 生成或编辑 | **98/100**，图片 48/50，视频 50/50 |
| 来源真实误报 | **0/70**，假材料检出 28/30 |
| HTTP 上传至完整结果落盘 | **最长 2.302 秒**，P95 2.167 秒，失败/超时 0 |
| 主流格式与证据链 | 22/22 完成 |
| 新人像来源池 | 209/210 正确，来源真实误报 0/90 |
| 文档局部 AI 修改专项 | **39/84 正确，11/28 误报，22/56 AI 编辑检出，尚未达标** |
| 代码和模型资源 | 156 项测试通过，9 个模型资源和必需上游文件校验通过 |

首轮新来源登记早于预测；随后集成注册与启动预热修复使用同一批材料复测，未因测试结果重训权重或重选阈值。专项人像与混合100共享部分保留池，不能合并成多轮独立盲测。0/70 的误报区间上界约5.20%，不能推定真实业务总体误报率为0。详见[验证报告](docs/VALIDATION.md)。

## 评委快速开始

建议下载 [v1.8 研究预览附件](https://github.com/God-Leelee/Lee/releases/tag/v1.8.0-preview.1)中的 **03 源码、05 视觉编码器、06 取证模型与源码**，解压到同一个父目录，得到一个 `Lee` 目录。不要形成 `Lee/Lee` 嵌套。

准备 **Windows、Python 3.12、兼容 NVIDIA 驱动和 GPU、约25GB可用磁盘**。参考机器为 RTX5060 Laptop8GB、RAM32GB、Python3.12.14、torch2.8.0+cu128。第二台机器、其他GPU、CPU回退尚待完整实机验证。

```powershell
cd Lee
powershell -NoProfile -ExecutionPolicy Bypass -File .\安装与初始化.ps1
```

首次安装需要联网安装固定 Python 依赖。两个权重附件包含必需模型和TruFor运行源码；普通体验不需要重新编译前端。若仅 `git clone`，安装器会按固定来源下载模型与上游源码。不会下载119张医疗原件。

安装结束后双击 **启动澄鉴.cmd**，等待约三分钟加载和预热，终端出现 `Application startup complete` 后打开 **http://127.0.0.1:8765/**。启动时间单列，单份五秒成绩针对服务就绪后的串行检测。保持终端运行，Ctrl+C停止。

## 重新生成演示结果

保持服务运行，在另一个 PowerShell 窗口执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/verify_assets.py
.\.venv\Scripts\python.exe -X utf8 scripts/make_demo_samples.py
.\.venv\Scripts\python.exe -X utf8 scripts/reproduce_demo.py --output runs/review-new-001
```

脚本重新上传并真实执行模型，核对输入SHA、配置、必需分支、原图掩码、实际PTS及JSON/CSV/HTML报告，输出目录必须全新。自制程序图片与短视频只验证功能，不带AI真假真值，不用于证明识别精度；出现高风险也不代表它们有已确认的AI生成标签。对照输出见[预期结果](docs/EXPECTED_RESULTS.md)。

## 复算100套研究测试

尊重各原发布源的研究使用条款，联网取得固定公开输入并重建已知编辑操作：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/restore_public100.py
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_live_15.py --suite mixed --version 1.8 --configuration-id v1.8-research-preview-003 --output review-public100-new-001
```

106个原始父源来自记录的发布者，100份测试输入和9份正常操作对照逐个校验SHA。来源受限时脚本会明确失败，不能换图后沿用旧成绩。测试会实际执行推理，失败、拒判、超时计入分母。初次下载需要网络，包中不是一套全离线操作系统与Python环境镜像。

## 技术与范围

CF224冻结表征与轻量MLP适配、CLIP/UFD、WaveRep、TruFor、SIFT与密集复制、连续窗口TCN、带框OCR及元数据共同构成主链路。图片生成和编辑分开建模，视频用条件证据树融合。源文件名、工具标签和真值掩码不传入模型。

支持JPEG、PNG、WebP、BMP、TIFF、AVIF、HEIC/HEIF，以及MP4、MOV、MKV、WebM、AVI。100MB、图片50MP、视频120秒/4K是接收上限，尚无全边界五秒保证。PDF、RAW、动态图像和音频转写未纳入。

风险支持人工核验，不等于保险欺诈概率。EXIF缺失不单独判假，视频未采样时间保持未知。运行数据位于本机 `data/cases/`。当前未部署公网。

## 材料与下一步

- [完整复现说明](docs/REPRODUCING.md)
- [技术架构与流程图](docs/ARCHITECTURE.md)
- [验证、错误与局限](docs/VALIDATION.md)
- [8分钟现场演示脚本](docs/DEMO_SCRIPT.md)
- [数据和公开范围](docs/DISTRIBUTION.md)
- [开源来源与原许可](THIRD_PARTY_NOTICES.md)

下一步优先补强医疗文档局部修改、传统拼接漏检及新工具来源。当前119张资源原件仅本地保留。SPAI和CO-SPY尚未完成同集比较，未进入发布链路。所有已有测试此后作为回归，正式达标需要新来源盲测。
