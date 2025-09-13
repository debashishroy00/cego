@echo off
echo ========================================
echo     CEGO Performance Benchmark
echo ========================================
echo.

cd /d "%~dp0.."

echo Running comprehensive benchmarks...
echo.

echo [1/4] Testing duplicate removal performance...
docker exec cego-dev python -c "
from src.optimizers.quick_wins import QuickWinsOptimizer
import time

opt = QuickWinsOptimizer()
sizes = [10, 50, 100, 500, 1000]

print('Size  | Reduction | Time')
print('------|-----------|------')
for size in sizes:
    docs = ['doc'] * (size // 2) + ['unique' + str(i) for i in range(size // 2)]
    start = time.time()
    result = opt.optimize_all(docs)
    elapsed = (time.time() - start) * 1000
    reduction = result['stats']['reduction']['token_reduction_pct']
    print(f'{size:5} | {reduction:8.1%%} | {elapsed:5.0f}ms')
"

echo.
echo [2/4] Testing API throughput...
docker exec cego-dev python -c "
import requests
import time
from concurrent.futures import ThreadPoolExecutor

def test_call(i):
    payload = {'query': f'q{i}', 'context_pool': ['d1', 'd2'], 'max_tokens': 100}
    start = time.time()
    r = requests.post('http://localhost:8000/optimize', json=payload)
    return time.time() - start

with ThreadPoolExecutor(max_workers=10) as executor:
    times = list(executor.map(test_call, range(50)))

print(f'Requests: 50')
print(f'Avg latency: {sum(times)/len(times)*1000:.1f}ms')
print(f'Throughput: {50/sum(times):.0f} req/s')
"

echo.
echo [3/4] Testing memory efficiency...
docker exec cego-dev python -c "
import tracemalloc
from src.optimizers.quick_wins import QuickWinsOptimizer

tracemalloc.start()
opt = QuickWinsOptimizer()

for size in [100, 500, 1000]:
    data = [f'Document {i}' * 10 for i in range(size)]
    result = opt.optimize_all(data)
    current, peak = tracemalloc.get_traced_memory()
    print(f'{size:4} docs: {peak/1024/1024:.1f}MB peak memory')

tracemalloc.stop()
"

echo.
echo [4/4] Comparison with baseline...
docker exec cego-dev python demo.py | findstr /I "reduction processing"

echo.
pause