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

固定运行资源见`models/runtime_assets.json`，`scripts/verify_assets.py`只读核对运行权重SHA、上游commit与配置。`scripts/check_public_tree.py`检查Git跟踪集合的公开范围。哈希证明文件一致，不提供真实性、模型准确率或商业授权证明。第三方代码在初始化时按固定commit克隆到被Git忽略的third_party目录，未修改其许可证。

实际保留的TruFor主许可和WaveRep许可要求：仅用于信息与非营利用途；禁止未经授权的工业或盈利活动；复制与修改保留原始权利通知；公开由该软件或其改编产生的结果时引用原作者。原始文件见`models/licenses/TruFor_LICENSE_OPEN.txt`、`third_party/TruFor/TruFor_train_test/LICENSE.txt`、`LICENSE_CMX.txt`及`third_party/WaveRep/LICENSE.md`、`demo/LICENSE.md`。主许可不因其中部分代码有MIT等宽松条款而消失。

本项目按非营利研究与评估用途保留这些条款，未取得商用授权。商业系统部署前需取得适当授权或替换受限组件；改名、重训练或蒸馏不能作为自动消除上游条款的依据。数据与编解码组件原始条款分别适用；PyAV代码BSD-3-Clause不等于所有FFmpeg组件授权。分发时保留原始许可，不能只提供此通知而省略对应文件。

`centered_temporal_tcn_v1`、训练流程、融合stacker、标签与样本管理、SIFT几何验证、报告和界面为本原型工程实现。TCN训练输入清单与checkpoint哈希见`runs/temporal_training.json`，融合checkpoint、开发样本ID、校准范围和阈值见`config.json`。D3共享CLIP适配和自行训练TCN不引用上游论文成绩作为本项目成绩。

RapidOCR包声明与模型来源归档：`models/licenses/RapidOCR_1.4.4_DECLARATION.json`。实际下载官方PyPI发行wheel校验SHA256为`971d7d5f223a7a808662229df1ef69893809d8457d834e6373d3854bc1782cbf`，三份已安装ONNX均与wheel内同名member一致。这里只确认发行包来源和包级Apache-2.0声明，没有臆造单独模型授权。

| 当前实际ONNX | SHA256 |
| --- | --- |
| `ch_PP-OCRv4_det_infer.onnx` | `d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9` |
| `ch_PP-OCRv4_rec_infer.onnx` | `48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b` |
| `ch_ppocr_mobile_v2.0_cls_infer.onnx` | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` |

## 项目训练的适配组件

本项目在冻结的公开编码器上训练了小型分类、校准与融合组件。训练流程、标签管理、复制匹配、报告和界面为本项目工程实现；自行训练的适配组件不改变上游编码器、数据集或取证模型的适用条款。

- [Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset/tree/787334f7857fa54f29027a7f09c30e895ad486ef)：仅在本地用于开发，原始图片不随仓库分发。
- [Community Forensics](https://github.com/JeongsooP/Community-Forensics)：MIT代码；CF224权重来自[OwensLab/commfor-model-224](https://huggingface.co/OwensLab/commfor-model-224)，固定revision为`26afc31e6b40c312c3fd42c05a758be62446215b`。本项目没有把论文成绩当作自身成绩。
- AIForge-Doc-v1/v2：仅按授权用于本地研究；原始文档与掩码不随仓库分发。其模型卡和数据描述存在许可差异时采用较严格的非商业范围。
- Gen-Video：仅在本地用于视频开发；原始媒体不随仓库分发，数据来源不构成商业授权。

TruFor源码包只保留运行需要的代码、配置和许可证，不包含重复训练权重、论文配图或示例图片；保留文件仍按固定上游commit校验。
