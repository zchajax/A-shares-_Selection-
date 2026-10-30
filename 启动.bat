@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=%~dp0venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] venv\Scripts\python.exe not found
    echo Create it with: py -3.12 -m venv venv
    echo Then: venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
"%PY%" -c "import dearpygui, pandas, akshare, numpy"
if errorlevel 1 (
    echo [ERROR] missing package. Install with:
    echo "%PY%" -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo Starting A-share stock picker...
"%PY%" main.py
pause
