# Windows 复现说明

配置 `v1.8-research-preview-003`。固定权重清单为 `models/runtime_assets.json`，必要依赖为 `requirements.lock`。本轮从精简交付目录启动后重新取得98/100、0/70误报、最长2.302秒。此验证使用本机已有依赖，不冒称第二台机器全新安装。

1. 下载03、05、06三个附件，解压到同一父目录形成Lee。
2. 安装Python3.12及兼容NVIDIA驱动，预留25GB磁盘。
3. 在Lee目录运行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\安装与初始化.ps1`。
4. 双击启动澄鉴.cmd，等待约3分钟模型加载和批量预热后访问本机地址。
5. 执行README中的make_demo_samples与reproduce_demo，使用全新runs输出目录。

已构建界面随源码提供。改前端时安装Node.js20+，在frontend目录执行npm ci和npm run build，或初始化时指定-RebuildFrontend。首次Python依赖安装需网络；现场已经完成安装时，启动不需要外部AI API。可自备匹配的完整wheelhouse并通过初始化参数-Wheelhouse指定，但本版未提供或验证全离线依赖安装镜像。

## 复算精度

`restore_public100.py` 访问原始发布者，校验106个父源，并按冻结操作参数重新生成传统编辑和正常对照。支持--check-only仅验证已存在输入；网络不支持Range时，可将原发布ZIP放到--archive-dir指定目录。脚本不会改阈值、筛掉错误或更换失败输入。

保持服务运行后执行evaluate_live_15.py，参数见README。新输出保存于benchmark_results/<输出名>，包括逐份预测、失败、耗时、置信区间、坐标和PTS核验。它重新推理，不读取旧分数冒充复现。相同权重在其他GPU上仍可能存在少量浮点差异，应同时比较分支、输入SHA和最终判定。

## 故障定位

| 现象 | 处理 |
|---|---|
| 127.0.0.1打不开 | 保持启动终端，等待Application startup complete，检查是否仍在模型加载 |
| Python3.14报依赖错误 | 使用py -3.12创建.venv |
| CUDA不可用 | 核对NVIDIA驱动及torch2.8.0+cu128，不承诺CPU五秒 |
| 权重缺失/哈希不符 | 解压两个权重附件并运行verify_assets.py，不跳过校验 |
| 源码下载网页受限 | 使用Release完整附件；测试原图另按原发布源获取 |
| 新复现目录已存在 | 更换全新runs目录，保留旧记录 |
| 长视频/大图/并发超过5秒 | 留存实际耗时，当前五秒结论仅覆盖已测串行范围 |

原件、案件和报告位于data/cases，Git忽略。源码公开树不包含用户材料或登录令牌。
