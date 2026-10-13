# 第三方方法、权重与运行库

| 来源 | 用途 | 条款 / 明示差异 |
| --- | --- | --- |
| [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) | CLIP分类头、官方输入和特征定义 | MIT代码；保留上游LICENSE。公开研究checkpoint来源与SHA256见models/sources.json。 |
| [OpenAI/CLIP](https://github.com/openai/CLIP)、[mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | OpenAI ViT-L/14编码器与场景文本向量 | MIT代码。实际OpenAI参数、QuickGELU、非归一化UFD特征。混合精度差异写入报告。 |
| [grip-unina/TruFor](https://github.com/grip-unina/TruFor) | 真实噪声/区域/可靠性/检测网络 | 主许可证仅允许信息与非营利用途；原始LICENSE.txt保留。本地LICENSE_CMX.txt为MIT，不能据此替代TruFor主条款。输入RGB/256.0保留，最长边768缩放、混合精度、可靠性过滤和坐标回映为改编。 |
| [grip-unina/WaveRep-SyntheticVideoDetection](https://github.com/grip-unina/WaveRep-SyntheticVideoDetection) | 发布的DINOv2-B/14 G4生成取证网络 | 根LICENSE.md与demo/LICENSE.md均限信息与非营利用途。完整checkpoint包含骨干及头；CenterCrop(504)、ImageNet归一化保留，实际PTS采样、混合精度和视频原生logit均值聚合为适配。 |
| [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | WaveRep骨干构建与OpenAI CLIP权重分发 | timm代码Apache-2.0；代码许可证不替代具体checkpoint适用条款。 |
| [Zig-HS/D3](https://github.com/Zig-HS/D3) | 连续特征一阶距离的二阶STD定义 | MIT。使用共享CLIP-L/14而非论文默认XCLIP-B/16，不引用论文成绩作为本项目成绩。 |
| [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR)、[PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、[官方PyPI 1.4.4发行](https://pypi.org/project/rapidocr-onnxruntime/1.4.4/) | 带框中文OCR真实推理 | 已归档包metadata的Apache-2.0声明与三份实际ONNX哈希，并与官方发行wheel内字节逐份匹配。该wheel没有独立许可证文件，包许可声明不自动补出独立权重许可；不声称模型商业授权已齐全。 |
| [opencv/opencv](https://github.com/opencv/opencv) | SIFT空间排除和仿射RANSAC、坐标恢复 | Apache-2.0；复制匹配及融合规则为本原型实现，不推断复制方向。 |
| [PyAV-Org/PyAV](https://github.com/PyAV-Org/PyAV) | FFmpeg视频轨道、PTS、解码与工程录像编码 | PyAV BSD-3-Clause；wheel附带FFmpeg依赖条款单独适用。 |
| [cccnju/Gen-Video](https://modelscope.cn/datasets/cccnju/Gen-Video)、MSRVTT及各生成工具源 | 实际TCN开发数据及公开保留集 | 媒体不随代码发布；原始archive member、SHA256、来源与开发分组逐条记录。数据来源不构成商业授权。 |
| [FastAPI](https://github.com/fastapi/fastapi)、[React](https://github.com/facebook/react)、[Vite](https://github.com/vitejs/vite)、[Lucide](https://github.com/lucide-icons/lucide) | 本机服务与工作台 | FastAPI/React/Vite代码MIT，Lucide为ISC；保留安装包原许可证，版本见对应锁文件。 |

GitHub评审版的固定资源见`models/runtime_assets.json`及`models/reproducibility_manifest.json`，`scripts/verify_assets.py`只读核对运行权重SHA、上游commit与配置。`scripts/check_public_tree.py`检查Git跟踪集合的公开范围。哈希证明文件一致，不提供真实性、模型准确率或商业授权证明。第三方代码在初始化时按固定commit克隆到被Git忽略的third_party目录，未修改其许可证。

实际保留的TruFor主许可和WaveRep许可要求：仅用于信息与非营利用途；禁止未经授权的工业或盈利活动；复制与修改保留原始权利通知；公开由该软件或其改编产生的结果时引用原作者。原始文件见`models/licenses/TruFor_LICENSE_OPEN.txt`、`third_party/TruFor/TruFor_train_test/LICENSE.txt`、`LICENSE_CMX.txt`及`third_party/WaveRep/LICENSE.md`、`demo/LICENSE.md`。主许可不因其中部分代码有MIT等宽松条款而消失。

本原型按竞赛非营利研究用途保留这些条款，未取得商用授权。商业理赔系统部署前需取得适当授权或替换受限组件；改名、重训练或蒸馏不能作为自动消除上游条款的依据。数据与编解码组件原始条款分别适用；PyAV代码BSD-3-Clause不等于所有FFmpeg组件授权。交付时保留原始许可，不能只交付此通知而省略对应文件。

`centered_temporal_tcn_v1`、训练流程、融合stacker、标签与样本管理、SIFT几何验证、报告和界面为本原型工程实现。TCN训练输入清单与checkpoint哈希见`runs/temporal_training.json`，融合checkpoint、开发样本ID、校准范围和阈值见`config.json`。D3共享CLIP适配和自行训练TCN不引用上游论文成绩作为本项目成绩。

RapidOCR包声明与模型来源归档：`models/licenses/RapidOCR_1.4.4_DECLARATION.json`。实际下载官方PyPI发行wheel校验SHA256为`971d7d5f223a7a808662229df1ef69893809d8457d834e6373d3854bc1782cbf`，三份已安装ONNX均与wheel内同名member一致。这里只确认发行包来源和包级Apache-2.0声明，没有臆造单独模型授权。

| 当前实际ONNX | SHA256 |
| --- | --- |
| `ch_PP-OCRv4_det_infer.onnx` | `d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9` |
| `ch_PP-OCRv4_rec_infer.onnx` | `48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b` |
| `ch_ppocr_mobile_v2.0_cls_infer.onnx` | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` |


1.2资源样本适配：复用已校验的OpenAI CLIP-L/14编码器，双视图逻辑回归头与训练／校准流程为本原型实现。资源来自用户授权的本机微信表ZIP导出，119张原图仅在本机用于分组开发与验证，不随源码包提供，不上传外部服务。代码、编码器、学习权重与源材料的适用条款分别保留；自行训练小分类头不改变上游编码器及其他取证模型条款。

1.3公开图片来源头：训练在冻结CLIP-L/14与WaveRep编码器的中心／全图特征及弱纹理描述上进行，逻辑回归、Platt校准和分组开发流程由本原型实现，没有实施新的大模型蒸馏。原始图片来自[Defactify_Image_Dataset固定revision](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset/tree/787334f7857fa54f29027a7f09c30e895ad486ef)，研究说明见[原始论文](https://arxiv.org/abs/2601.00553)。原始parquet字节、caption组隔离、源文件SHA与受限分片抽样范围记录在本机provenance；训练600张、校准100张、完整阈值开发100张，保留测试100张。原始图片及特征缓存不随源码包提供。上游预训练交叉未知，自训头不改变编码器、WaveRep或数据条款。

1.3视频扩充来源：[Gen-Video原始验证归档](https://modelscope.cn/datasets/cccnju/Gen-Video/resolve/master/GenVideo-Val.zip)。以目录身份、CRC、HTTP ETag、原始SHA和父源排除规则固定100条新视频；素材来源工具名为数据集标签，未取得逐条生成日志。重复性能回归和混合样例与专项测试的相关性完整保留，不声称300个独立业务案件。数据原件只在本机使用，保留原始出处与限制。


## 1.8 更新的来源

Community Forensics： https://github.com/JeongsooP/Community-Forensics ，代码 MIT。CF224 权重来自 https://huggingface.co/OwensLab/commfor-model-224 ，固定 revision `26afc31e6b40c312c3fd42c05a758be62446215b`，MIT 模型卡和 LICENSE 原文位于 licenses/。本项目在其冻结编码器上训练特征适配头，未把论文成绩当成本项目成绩。

SPAI（https://github.com/mever-team/spai）和 CO-SPY（https://github.com/Megum1/CO-SPY）仅完成候选资料及资源准备，未进入本版推理，也未宣称完成同集效果对比。AIForge-Doc-v1/v2 仅用于授权的本地研究。其模型卡 CC BY-NC-SA 4.0 与 dataset_info 的 CC BY 4.0 存在冲突，采用较严格范围；原始文档与掩码均不在公开交付包。

TruFor 源码包仅保留运行所需的 lib、配置与许可证，不包含重复的训练骨干权重、训练文件列表、论文配图和示例图片。每个保留文件仍按原始固定 commit 的 SHA-256 核验。
