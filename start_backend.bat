@echo off
setlocal enabledelayedexpansion
title CEGO Backend Launcher

REM === Settings (edit if needed) ===
set APP=backend.api.main:app
set DEFAULT_PORT=8003
set HOST=127.0.0.1

REM === Pick port (optional arg) ===
if "%~1"=="" (
  set /p PORT=Enter port [%DEFAULT_PORT%]: 
  if "!PORT!"=="" set PORT=%DEFAULT_PORT%
) else (
  set PORT=%~1
)

echo Using port %PORT%

REM === Activate virtual env if found ===
set VENV_DIR=venv
if exist "%VENV_DIR%\Scripts\activate.bat" (
  echo Activating venv: %VENV_DIR%
  call "%VENV_DIR%\Scripts\activate.bat"
) else (
  echo [Info] No venv at "%VENV_DIR%". Using system Python.
)

REM === Check if port is in use ===
set PID=
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do (
  set PID=%%a
)
if defined PID (
  echo [Warn] Port %PORT% is in use by PID %PID%.
  choice /M "Kill this process?"
  if errorlevel 2 (
    echo Choose a different port when re-running, e.g.: start_backend.bat 8010
    goto :START_SERVER
  ) else (
    echo Killing PID %PID% ...
    taskkill /F /PID %PID% >nul 2>&1
    timeout /t 1 >nul
  )
)

:START_SERVER
echo Starting uvicorn on http://%HOST%:%PORT% ...
REM Use python -m to avoid PATH issues
python -m uvicorn %APP% --host %HOST% --port %PORT% --reload
if errorlevel 1 (
  echo.
  echo [Error] uvicorn failed to start.
  echo Tips:
  echo   - Try another port: start_backend.bat 8010
  echo   - Run as Administrator if you still see WinError 10013
  echo   - Disable VPN/Firewall temporarily or allow Python through firewall
  echo   - Ensure module path is correct: %APP%
  echo   - Confirm you're in repo root (where "backend\" folder exists)
  echo.
  pause
)
endlocal
