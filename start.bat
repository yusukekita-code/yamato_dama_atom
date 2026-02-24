@echo off
chcp 65001 >nul
title AI トリオ協議システム
cd /d "%~dp0ai_panel"

py launch.py 2>nul || python launch.py

pause
