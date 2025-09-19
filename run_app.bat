@echo off
echo AI Career Path Recommender - BrainWonders Internship Project
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "career_env" (
    echo Creating virtual environment...
    python -m venv career_env
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call career_env\Scripts\activate.bat

REM Check if dependencies are installed
career_env\Scripts\python.exe -c "import PyQt6" 2>nul
if %errorlevel% neq 0 (
    echo Installing dependencies...
    career_env\Scripts\pip.exe install -r requirements.txt
    echo.
)

REM Run the application
echo Starting Career Recommender Application...
echo.
career_env\Scripts\python.exe main_lite.py

echo.
echo Application closed.
pause