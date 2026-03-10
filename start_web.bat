@echo off
chcp 65001 >nul
echo ====================================================
echo    中文文本智能纠错系统 - Web 启动脚本
echo ====================================================
echo.
echo 正在激活虚拟环境...
call venv\Scripts\activate.bat
echo.
echo 正在启动 Web 应用...
echo.
python web_app_gradio.py
pause

