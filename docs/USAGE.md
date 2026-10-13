# 使用说明

## 新建核验

1. 启动本地服务并打开 `http://127.0.0.1:8765/`。
2. 选择“新建材料核验”。
3. 上传一张图片或一段视频。
4. 等待全部已启用分支结束。
5. 查看材料风险分、证据列表、可疑区域和运行记录。

结果页中的分数来自多个证据分支。某个分支未运行、超时或资源缺失时，界面和报告会保留状态，不会把它当作零风险证据。

## 案件记录

每次分析会在 `data/cases/<case_id>/` 中保存：

- 输入文件及 SHA-256
- 运行配置和分支状态
- JSON 结构化报告
- HTML 阅读报告
- 热力图、掩码和关键帧等派生证据

删除案件前请先备份需要保留的报告。仓库更新不会自动上传或同步案件目录。

## 视频结果

视频报告中的时间位置来自容器实际 PTS。系统只对采样帧和分析窗口给出证据；未采样区间保持未知。短时编辑可能落在采样间隔内，因此需要结合原视频和其他材料复核。

## 常见问题

### 网页打不开

确认启动窗口仍在运行，并访问 `http://127.0.0.1:8765/`。如果端口被占用，关闭旧进程或使用：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\run_local.py --port 8766
```

### CUDA 不可用

运行：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

若输出 `False`，检查 NVIDIA 驱动及是否安装了 CUDA 版 PyTorch。

### 模型资源缺失

运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\fetch_models.py
.\.venv\Scripts\python.exe -X utf8 scripts\verify_assets.py
```

脚本会按资源锁文件下载或验证；哈希不符时不会继续启动。

### 重新编译前端

```powershell
cd frontend
npm.cmd ci
npm.cmd run build
```

构建产物写入 `frontend/dist`。
