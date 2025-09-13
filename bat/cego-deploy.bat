@echo off
echo ========================================
echo     CEGO Production Deployment
echo ========================================
echo.

cd /d "%~dp0.."

echo Select deployment target:
echo 1. Development (port 8001)
echo 2. Production (port 8002)
echo 3. Both
echo.

set /p choice=Enter your choice (1-3): 

if %choice%==1 (
    docker-compose up -d cego-dev
    echo Development deployed on http://localhost:8001
) else if %choice%==2 (
    docker-compose up -d cego-prod
    echo Production deployed on http://localhost:8002
) else if %choice%==3 (
    docker-compose up -d
    echo All services deployed
) else (
    echo Invalid choice
)

echo.
pause