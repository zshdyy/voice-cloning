@echo off
REM ============================================
REM 智能性别语音转换系统 - GUI 启动脚本 (CMD版)
REM ============================================
REM 用途：一键启动图形界面（Windows CMD/Batch）
REM 平台：Windows Command Prompt / Batch

chcp 65001 >nul
setlocal enabledelayedexpansion

cls
color 0B
echo.
echo ============================================
echo Voice Gender Changer - GUI Launcher
echo ============================================
echo.

REM 检查虚拟环境
if not exist "voice_env_clean" (
    echo ERROR: Virtual environment not found.
    echo Run: python -m venv voice_env_clean
    echo Then: .\voice_env_clean\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo OK: Virtual environment detected

REM 激活虚拟环境
echo Activating virtual environment...
call voice_env_clean\Scripts\activate.bat

REM 设置 Qt 插件路径
set "QtPluginPath=%CD%\voice_env_clean\Lib\site-packages\PyQt5\Qt5\plugins\platforms"
if not exist "%QtPluginPath%" (
    echo ERROR: Qt plugin path not found.
    echo Try: pip install --upgrade PyQt5
    pause
    exit /b 1
)

set QT_QPA_PLATFORM_PLUGIN_PATH=%QtPluginPath%
echo OK: Qt plugin path configured

REM 检查 voice_gui.py
if not exist "voice_gui.py" (
    echo ERROR: voice_gui.py not found.
    pause
    exit /b 1
)

echo OK: GUI script detected
echo.
echo Launching GUI...
echo - Drag a WAV file into the window
echo - Click Start Convert to process
echo - Use F0 chart to view/save
echo.

REM 启动 GUI
"%CD%\voice_env_clean\Scripts\python.exe" voice_gui.py

if errorlevel 1 (
    echo ERROR: Launch failed.
    pause
    exit /b 1
)

echo.
echo GUI closed.
pause
