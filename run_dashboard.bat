@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project virtual environment not found at .venv\Scripts\python.exe
    echo Create it and install the project requirements before starting the dashboard.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run "app\dashboard.py"
pause
