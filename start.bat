@echo off
echo ========================================
echo  AI Trio Discussion System
echo ========================================
echo.

echo [1] Moving to ai_panel directory...
cd /d "%~dp0ai_panel"
echo     %CD%
echo.

echo [2] Checking .env file...
if not exist ".env" (
    echo     ERROR: .env file not found!
    echo     Please copy .env.example to .env and set your API keys.
    goto :end
)
echo     .env found OK
echo.

echo [3] Checking Python...
py --version
if %errorlevel% neq 0 (
    echo     ERROR: py command failed. Exit code: %errorlevel%
    goto :end
)
echo.

echo [4] Launching server...
py launch.py

echo.
echo Server has stopped. Exit code: %errorlevel%

:end
echo.
echo Press any key to close this window...
pause > nul
