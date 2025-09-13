@echo off
echo ========================================
echo      Starting CEGO System
echo ========================================
echo.

cd /d "%~dp0.."

echo Checking for port conflicts...
netstat -ano | findstr :8001 >nul
if %errorlevel%==0 (
    echo WARNING: Port 8001 may be in use
    echo.
)

echo Stopping any existing CEGO containers...
docker-compose down

echo.
echo Starting CEGO development server...
docker-compose up -d cego-dev

echo.
echo Waiting for services to be healthy...
timeout /t 5 /nobreak > nul

echo.
echo Checking service status...
docker-compose ps

echo.
echo ========================================
echo CEGO is ready!
echo ========================================
echo API: http://localhost:8001
echo Health: http://localhost:8001/health
echo Docs: http://localhost:8001/docs
echo ========================================
pause