@echo off
echo ========================================
echo       CEGO Demo & Examples
echo ========================================
echo.

cd /d "%~dp0.."

echo Running main demo...
docker exec cego-dev python demo.py

echo.
echo ========================================
echo Running enhanced test scenarios...
echo ========================================
docker exec cego-dev python src/enhanced_test.py 2>nul || (
    echo Enhanced test not found, creating...
    docker cp enhanced_test.py cego-dev:/app/src/enhanced_test.py
    docker exec cego-dev python src/enhanced_test.py
)

echo.
pause