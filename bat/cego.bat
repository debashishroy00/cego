@echo off
:menu
cls
echo ========================================
echo       CEGO Docker Management
echo    Context Entropy Gradient Optimizer
echo ========================================
echo.
echo 1. Start development server
echo 2. Stop all services
echo 3. Run demo
echo 4. Run tests
echo 5. View logs
echo 6. Health check
echo 7. Performance benchmark
echo 8. Full reset (CAUTION!)
echo 9. Exit
echo.

set /p choice=Select option (1-9): 

if %choice%==1 call "%~dp0cego-start.bat"
if %choice%==2 call "%~dp0cego-stop.bat"
if %choice%==3 call "%~dp0cego-demo.bat"
if %choice%==4 call "%~dp0cego-test.bat"
if %choice%==5 call "%~dp0cego-logs.bat"
if %choice%==6 call "%~dp0cego-health.bat"
if %choice%==7 call "%~dp0cego-benchmark.bat"
if %choice%==8 call "%~dp0cego-reset.bat"
if %choice%==9 exit

goto menu