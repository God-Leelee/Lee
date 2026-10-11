# 澄鉴 CLEARTRACE

图片与视频材料反欺诈研究原型 · **1.5.0 评审源码版**

这是众安 1024 AI Coding 竞赛项目的源码与本地复现入口。评委可以下载工程，在本机运行玻璃风格的取证工作台，上传图片或视频，查看分支证据、区域掩码、实际视频时间戳及核验报告。

![澄鉴本地取证工作台，截图仅含程序生成的工程样例](docs/images/workbench.png)

**当前尚未全项达标。** 2026-10-11 的既有公开研究集回归为 97/100、来源真实误报 0/70、假材料召回 27/30；图片 47/50。视频完整检测出现超过 5 秒的样本，微信资源包域的效果仍有明显差距。这些结果属于原配置的研究证据，不是本次源码发布版重新取得的独立准确率成绩。详情见 [验收状态](docs/VALIDATION.md)。

## 评委快速开始

已测试的参考机器为 **Windows、NVIDIA RTX 5060 Laptop 8GB、约 32GB 内存、Python 3.12、PyTorch 2.8.0 + CUDA 12.8**。准备 Git、Python 3.12、Node.js 20+ 和兼容的 NVIDIA 驱动，建议预留至少 15GB 磁盘；模型初始化需要联网。其他显卡、CPU、macOS 和 Linux 尚未完成复现验证。

```powershell
git clone https://github.com/God-Leelee/Lee.git
cd Lee
powershell -NoProfile -ExecutionPolicy Bypass -File .\安装与初始化.ps1
```

安装完成后双击 **`启动澄鉴.cmd`**，等待终端提示服务启动，再打开：

**http://127.0.0.1:8765/**

此地址属于运行工程的那台电脑。首次模型加载可能需要约一分钟；服务仅监听本机。首次安装结束后，启动器会校验固定资源并启用 Hugging Face 离线模式。

## 演示与结果核验

保持服务运行，在另一个 PowerShell 窗口执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/make_demo_samples.py
.\.venv\Scripts\python.exe -X utf8 scripts/reproduce_demo.py --output runs/review-demo-001
```

演示样例由代码在本机生成，无主办方或第三方原图，包含工程图片、已知复制操作和短视频。脚本会真正调用模型，核对原件 SHA-256、必需分支、掩码尺寸、视频 PTS 和 JSON/CSV/HTML 报告，输出新的实测记录。**这些程序绘制的样例只验证运行与证据链，不用于证明 AI 检测精度，也不要求出现预设高低风险判定。**

支持上传 JPEG、PNG、WebP、BMP、TIFF、AVIF、HEIC/HEIF，以及 MP4、MOV、MKV、WebM、AVI。单文件 100MB、50MP 图片、120 秒/4K 视频是接收上限，尚未证明全范围均能在五秒内完成。PDF、相机 RAW、动画和音频转写未包含在已支持能力中。

## 原型如何工作

```text
原件保留与 SHA-256
        ↓
图片安全解码 / 视频实际 PTS 采样
        ↓
CLIP + WaveRep 来源特征 ── TruFor 区域取证 + SIFT 复制证据
        ↓                         ↓
已训练时序 TCN / 图片和视频适配头 / OCR 图文核验 / 元数据
        ↓
训练融合、独立划分校准与阈值
        ↓
材料风险、区域与时间证据、JSON / CSV / HTML 报告
```

页面显示的概率只适用于所用公开参考域，不能解释为保险欺诈概率。EXIF 缺失不单独证明伪造；视频未采样时刻保持未知。

## 仓库内容与资源获取

| 内容 | 交付方式 |
|---|---|
| FastAPI 后端、React 前端、训练/测试脚本 | 当前仓库 |
| 当前主链路的小型图片头、视频来源头和时序权重 | `models/candidates/`，固定哈希 |
| CLIP、TruFor、WaveRep、UFD 上游权重 | 安装时从记录的原始来源获取并校验 |
| 上游代码与许可证 | 固定 commit 获取；许可证原文同时保留在 `licenses/` |
| 无第三方原图的演示输入 | `scripts/make_demo_samples.py` 本地生成 |
| 主办方微信图片、公开原始媒体、案件原件、特征缓存 | 不公开上传；完整历史复现需要另行取得授权输入 |

公开版关闭了含私有样本支持向量的微信辅助来源头，主图片/视频融合权重和阈值保持原值；配置 ID 加了 `-github-review` 后缀以区分。它不能冒充原配置的逐分支完整复现。安装脚本不会自动下载微信资源。

## 更多资料

- [完整本地复现说明](docs/REPRODUCING.md)
- [技术方案与证据边界](docs/ARCHITECTURE.md)
- [当前验收状态与缺口](docs/VALIDATION.md)
- [数据、模型和公开范围](docs/DISTRIBUTION.md)
- [开源来源与上游使用条件](THIRD_PARTY_NOTICES.md)
- [发布前的原配置研究记录](docs/RESEARCH_HISTORY.md)

运行后原件、案件和报告保存在本机 `data/cases/`，不提交到 GitHub。当前源码用于本机研究与竞赛评审，尚未完成面向公网的账号隔离、限流和持久队列建设。

项目自有代码保留权利，并授权评委进行非商业竞赛评审；当前未套用覆盖全部组件的 MIT 许可证。TruFor 和 WaveRep 的信息/非营利限制继续适用，详见 [LICENSE](LICENSE)。
