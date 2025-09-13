@echo off
echo ========================================
echo    WARNING: Full CEGO System Reset
echo ========================================
echo This will DELETE all data and rebuild everything!
echo.

set /p confirm=Are you sure? Type YES to continue: 

if not "%confirm%"=="YES" (
    echo Reset cancelled.
    pause
    exit /b
)

cd /d "%~dp0.."

echo Stopping all containers...
docker-compose down -v

echo.
echo Removing old images...
docker rmi cego-cego-dev cego-cego-prod cego-cego-test 2>nul

echo.
echo Rebuilding images...
docker-compose build --no-cache

echo.
echo Starting fresh...
docker-compose up -d cego-dev

echo.
echo Running initial test...
timeout /t 10 /nobreak > nul
docker exec cego-dev python demo.py

echo.
echo System reset complete!
pause