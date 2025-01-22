@echo off

:: Check if Python 3.9.13 is installed
for /f "tokens=2" %%a in ('python --version 2^>^&1') do set python_version=%%a
if not "%python_version%"=="3.9.13" (
    echo Python 3.9.13 is not installed.
    echo Please install Python 3.9.13 from the Microsoft Store:
    start https://apps.microsoft.com/detail/python-3913/9PJPW5LDXLZ5
	pause
    exit /b 1
)

:: Install required Python packages
echo Installing required Python packages...
pip install opencv-python dlib pyserial
pause

:: Verify installation
python -c "import cv2, dlib, serial" 2>nul
if %errorlevel% equ 0 (
    echo All dependencies installed successfully!
    pause
) else (
    echo Error: Failed to verify all dependencies.
	pause
    exit /b 1
)
pause
