@echo off
chcp 65001 >nul
title AI トリオ協議システム

echo.
echo ==========================================
echo   AI トリオ協議システム
echo ==========================================
echo.

cd /d "%~dp0ai_panel"

:: .env ファイルの存在チェック
if not exist ".env" (
    echo [エラー] .env ファイルが見つかりません。
    echo.
    echo .env.example をコピーして .env を作成し、
    echo APIキーを記入してください。
    echo.
    goto :end
)

:: Python で起動
py launch.py
if %errorlevel% neq 0 (
    python launch.py
)

:end
echo.
echo ウィンドウを閉じると終了します。
pause
