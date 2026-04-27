@echo off
echo Starting Photo Rename Tool Backend Server...
cd /d "%~dp0"
cd backend
echo Working directory: %cd%
echo.
python run.py
pause