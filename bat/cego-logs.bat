@echo off
echo ========================================
echo          CEGO Logs
echo ========================================
echo.

cd /d "%~dp0.."

echo Select which logs to view:
echo 1. CEGO development logs
echo 2. CEGO production logs
echo 3. All logs
echo 4. Follow dev logs (real-time)
echo 5. Error logs only
echo.

set /p choice=Enter your choice (1-5): 

if %choice%==1 (
    docker-compose logs --tail=100 cego-dev
) else if %choice%==2 (
    docker-compose logs --tail=100 cego-prod 2>nul || echo Production not running
) else if %choice%==3 (
    docker-compose logs --tail=50
) else if %choice%==4 (
    docker-compose logs -f cego-dev
) else if %choice%==5 (
    docker-compose logs --tail=100 | findstr /I "error exception fail"
) else (
    echo Invalid choice
)

pause