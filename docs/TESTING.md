# 测试说明

## 自动化测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试覆盖配置校验、格式解析、风险融合、案件存储、API 契约、视频时间戳和错误处理。模型资源未安装时，部分测试使用受控替身验证应用逻辑；这类测试不能替代真实模型推理。

## 资源校验

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\verify_assets.py
```

该命令根据 `models/runtime_assets.json` 核对必需文件、上游提交和 SHA-256。

## 端到端回放

先启动服务，再运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\make_demo_samples.py
.\.venv\Scripts\python.exe -X utf8 scripts\reproduce_demo.py --base http://127.0.0.1:8765 --output runs/demo-new-001
```

回放会提交公开可再分发的程序生成样例，并检查：

- 请求完成且案件可重新读取
- 必需模型分支确实运行
- 报告包含输入哈希和配置标识
- 图像区域坐标可映射回原图
- 视频结果包含实际时间戳
- JSON、CSV 或 HTML 等预期输出存在

演示样例只用于验证工程链路，不提供真实/生成标签，也不代表检测准确率。

## 性能测量

`scripts/benchmark.py` 可记录接口完整耗时。测量时应分别记录冷启动、热态重复请求、输入解码、模型推理和报告落盘，避免用单个内部算子耗时代替用户实际等待时间。
