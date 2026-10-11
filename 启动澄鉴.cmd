@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 请先按README运行 安装与初始化.ps1。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 scripts/run_local.py
if errorlevel 1 echo 启动失败，请检查上方提示。若8765端口已占用，请先停止旧服务。
pause
