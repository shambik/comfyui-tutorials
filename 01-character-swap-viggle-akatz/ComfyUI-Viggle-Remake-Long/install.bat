@echo off
rem Double-click to install the Viggle Remake Long nodes into your ComfyUI (it does not restart ComfyUI).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
