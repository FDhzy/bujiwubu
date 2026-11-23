@echo off
cd /d "%~dp0"
if exist "%LocalAppData%\Programs\Python\Python310\python.exe" (
  "%LocalAppData%\Programs\Python\Python310\python.exe" "%~dp0shoe_photo_demo.py"
) else (
  python "%~dp0shoe_photo_demo.py"
)
if errorlevel 1 pause
