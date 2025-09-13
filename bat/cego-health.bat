@echo off
echo ========================================
echo        CEGO Health Check
echo ========================================
echo.

cd /d "%~dp0.."

echo Checking container status...
docker-compose ps

echo.
echo Testing API health endpoint...
curl -s http://localhost:8001/health || echo API health check failed

echo.
echo Testing optimization endpoint...
curl -X POST http://localhost:8001/optimize ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"test\", \"context_pool\": [\"doc1\", \"doc1\", \"doc2\"], \"max_tokens\": 1000}" ^
  -s -o nul -w "Optimization endpoint: HTTP %%{http_code}\n" || echo Optimization check failed

echo.
echo Checking memory usage...
docker stats cego-dev --no-stream

echo.
echo Quick performance test...
docker exec cego-dev python -c "from src.optimizers.quick_wins import QuickWinsOptimizer; import time; opt = QuickWinsOptimizer(); data = ['test'] * 100; start = time.time(); opt.optimize_all(data); elapsed = (time.time() - start) * 1000; print(f'100 doc optimization: {elapsed:.1f}ms')"

echo.
pause