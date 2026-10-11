# 技术方案与反欺诈原理

系统把材料来源、局部编辑、图文与元数据疑点分开计算，再融合能够验证的视觉证据。AI生成标签不等于保险欺诈真值，最终报告供人工复核。

| 监测点 | 已实现的方法 | 证据及边界 |
|---|---|---|
| AI图片 | 冻结CLIP中心/全图、WaveRep、多视图监督适配头 | 来源风险；不能解释为每个像素均伪造 |
| AI视频 | 实际PTS采样、两个连续窗口、已训练TCN、关键帧来源头 | 真实采样时间点和窗口线索；未采样间隔未知 |
| 传统篡改 | TruFor噪声/区域/可靠性、SIFT几何复制证据 | 原图像素掩码、匹配几何；真实重绘域仍待验证 |
| 多模态核验 | CLIP场景 + RapidOCR框/文字/可信度 + 字段规范化 | 显式日期/金额等冲突，低置信度字段不制造矛盾 |
| 元数据 | EXIF、容器和音视频轨道 | 辅助来源提示；缺失EXIF不能独立证明伪造 |
| Ensemble | 图片训练融合、视频证据树、独立划分校准与阈值 | 参考域风险和分支状态；缺失主证据不判为真实 |
| 统一平台 | FastAPI、React、文件原件与证据保存 | 案件、区域预览、PTS帧、JSON/CSV/HTML导出 |

主要开源来源为 [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)、[WaveRep](https://github.com/grip-unina/WaveRep-SyntheticVideoDetection)、[TruFor](https://github.com/grip-unina/TruFor)、[D3](https://github.com/Zig-HS/D3)、[RapidOCR](https://github.com/RapidAI/RapidOCR)。模型哈希与上游commit锁定在 `models/runtime_assets.json`。

数据按来源/父原件分组，训练、候选选择、校准、阈值和冻结测试分离。推理请求只含中性文件名和空描述，评估标签、工具名与真值掩码不得作为模型输入。失败、超时、拒判保留在统计分母。已多次使用的保留数据用于回归后，后续达标需要新来源冻结测试。

原型采用预训练编码器上的监督适配与融合训练，没有声称完成新的教师—学生知识蒸馏。公开版关闭私有支持向量辅助头，其余主视觉链路保持原模型与阈值。
