@echo off
chcp 65001 >nul
echo ========================================
echo   Photo Rename Tool 启动选项
echo ========================================
echo.
echo   [1] 标准模式（需要大内存/GPU）
echo   [2] 优化模式（4GB显存推荐）
echo   [3] 低内存模式（解决页面文件太小）← 推荐
echo   [4] 退出
echo.
set /p choice=请选择启动模式 (1-4): 

if "%choice%"=="1" goto standard
if "%choice%"=="2" goto optimized
if "%choice%"=="3" goto lowmem
if "%choice%"=="4" goto end
goto end

:standard
echo.
echo 启动标准模式...
cd /d "%~dp0"
cd backend
python run.py
pause
goto end

:optimized
echo.
echo 启动优化模式...
cd /d "%~dp0"
cd backend
python run_optimized.py
pause
goto end

:lowmem
echo.
echo ========================================
echo   低内存模式（解决页面文件太小）
echo ========================================
echo.
echo 正在配置内存优化...
echo - 限制PyTorch线程数
echo - 使用轻量模型 (ViT-GPT2)
echo - 默认使用CPU避免GPU OOM
echo.
cd /d "%~dp0"
cd backend
python run_low_memory.py
pause
goto end

:end