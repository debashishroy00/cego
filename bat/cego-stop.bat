@echo off
echo ========================================
echo       Stopping CEGO System
echo ========================================
echo.

cd /d "%~dp0.."

docker-compose down

echo.
echo All CEGO services stopped.
pause