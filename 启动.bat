@echo off
chcp 65001 >nul
cd /d %~dp0
set PYENV=C:\Users\chacezhang\.workbuddy\binaries\python\envs\gupiao\Scripts\python.exe
if not exist "%PYENV%" (
    echo [ERROR] venv not found, check README
    pause
    exit /b 1
)
echo Starting A-share stock picker...
"%PYENV%" main.py
pause
