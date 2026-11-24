@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run Setup-Windows.ps1 from PowerShell.
  exit /b 1
)
".venv\Scripts\python.exe" launch_desktop.py

