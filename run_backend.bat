@echo off
echo Starting UniAssist AI Backend Server...
cd /d "%~dp0backend"
call venv\Scripts\activate.bat
venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
pause
