# ANNEX H: REPRODUCIBILITY

## 1. ENVIRONMENT SNAPSHOT

- **Python Version**: 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]
- **Platform**: Windows-10-10.0.26100-SP0
- **Processor**: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD
- **Timestamp**: 2025-09-14T18:15:44.411809

## 2. PACKAGE VERSIONS

- **numpy**: 2.3.1
- **pandas**: 2.2.3
- **scikit-learn**: 1.6.1
- **matplotlib**: 3.10.6
- **jinja2**: 3.1.6
- **requests**: 2.32.5

## 3. CONFIGURATION FILES

Configuration saved to: `configs_snapshot/config.json`

### Key Parameters
- **Datasets**: 6
- **Repetitions**: 3
- **Max Items**: 50
- **Timeout**: 30.0s

## 4. REPRODUCTION STEPS

```bash
# 1. Install dependencies
pip install -r cego_bench/requirements.txt

# 2. Run benchmark
python -m cego_bench.runners.run_bench --config configs/default.yaml

# 3. Generate patent report
python -m cego_bench.reporting.patent_report
```

## 5. GIT INFORMATION

- **Commit Hash**: d5c83fd0344098bbd3295f4d07f6a908e7579199
- **Branch**: main
