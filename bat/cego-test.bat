@echo off
echo ========================================
echo         CEGO Test Suite
echo ========================================
echo.

cd /d "%~dp0.."

echo Select test type:
echo 1. Quick unit tests
echo 2. Full test suite
echo 3. Performance tests only
echo 4. Golden query validation
echo 5. Coverage report
echo.

set /p choice=Enter your choice (1-5): 

if %choice%==1 (
    echo Running quick unit tests...
    docker exec cego-dev pytest tests/unit/ -v
) else if %choice%==2 (
    echo Running full test suite...
    docker exec cego-dev pytest tests/ -v
) else if %choice%==3 (
    echo Running performance benchmarks...
    docker exec cego-dev python benchmarks/run_benchmarks.py
) else if %choice%==4 (
    echo Running golden query validation...
    docker exec cego-dev python -c "from src.validators.golden_queries import GoldenQueryValidator; from src.api.quick_wins_api import QuickWinsAPI; v = GoldenQueryValidator(); api = QuickWinsAPI(); r = v.validate_optimizer(api.optimize); print(f'Pass rate: {r[\"summary\"][\"passed\"]}/{r[\"summary\"][\"total\"]}')"
) else if %choice%==5 (
    echo Running tests with coverage...
    docker exec cego-dev pytest tests/ --cov=src --cov-report=term-missing
) else (
    echo Invalid choice
)

echo.
pause