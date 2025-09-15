# CEGO Benchmark Harness v1.0

A comprehensive benchmark harness for evaluating CEGO (Context Efficient Generation Optimization) algorithms on reduction percentage, semantic retention, latency, and junk handling across multiple domains.

## Features

- **Multi-domain Testing**: Insurance, SDLC, mixed intent, noisy data, duplicates, and formatting edge cases
- **Dual Optimizer Support**: Pattern (conservative) and Entropy (aggressive) optimization endpoints
- **Comprehensive Metrics**: Token reduction, semantic retention, latency, junk handling, intent preservation
- **Multiple Scoring Methods**: Embedding-based, TF-IDF, and ROUGE similarity scoring
- **Patent-Ready Reports**: Self-contained HTML reports with charts, statistics, and acceptance gate results
- **Acceptance Gates**: Automated pass/fail criteria with suggested parameter tweaks

## Quick Start

### Prerequisites

```bash
pip install requests numpy scipy scikit-learn pandas matplotlib jinja2 tqdm pyyaml
# Optional: pip install sentence-transformers  # For embedding scoring
```

### Basic Usage

```bash
# Run default benchmark
python -m cego_bench.runners.run_bench --config configs/default.yaml

# Run stress test
python -m cego_bench.runners.run_bench --config configs/stress.yaml

# Single dataset test
python -m cego_bench.runners.run_bench --dataset cego_bench/datasets/insurance.jsonl --repeat 1

# Skip embeddings (TF-IDF fallback)
python -m cego_bench.runners.run_bench --config configs/default.yaml --no-embed
```

### Configuration

Edit `configs/default.yaml` or `configs/stress.yaml`:

```yaml
endpoints:
  pattern: "http://127.0.0.1:8003/optimize/pattern"
  entropy: "http://127.0.0.1:8003/optimize/entropy"

runs:
  repeat: 3
  max_items: 50

acceptance_gates:
  entropy:
    min_reduction_median: 60.0
    min_retention_mean: 0.92
  pattern:
    min_reduction_median: 20.0
    min_retention_mean: 0.90
```

## Architecture

```
cego_bench/
├── runners/           # Core benchmark logic
│   ├── loaders.py     # JSONL dataset loading
│   ├── adapters.py    # HTTP endpoint adapters
│   ├── metrics.py     # Metrics calculation
│   └── run_bench.py   # Main CLI orchestrator
├── scorers/           # Similarity scoring algorithms
│   ├── embed_scorer.py   # Sentence embeddings
│   ├── tfidf_scorer.py   # TF-IDF similarity
│   └── rouge_scorer.py   # ROUGE lexical overlap
├── reporting/         # Results aggregation and reports
│   ├── aggregate.py   # Statistical analysis
│   └── html_report.py # HTML report generation
├── datasets/          # Test datasets (JSONL)
├── configs/           # Configuration files
└── utils/             # Timing and hashing utilities
```

## Dataset Format

Each dataset is a JSONL file with test cases:

```json
{
  "id": "test_001",
  "query": "underwriting analysis for cyber liability",
  "items": [
    {
      "id": "a",
      "text": "Cyber liability assessment...",
      "domain_hint": "insurance",
      "is_junk_gt": false
    }
  ],
  "gold": {
    "kept_item_ids": ["a", "b"],
    "intent_tags": ["insurance", "cyber"]
  },
  "notes": "Example test case"
}
```

## Metrics

### Core Metrics
- **Token Reduction**: Percentage of tokens removed
- **Semantic Retention**: Similarity between original and optimized content
- **Latency**: Response time in milliseconds
- **Confidence**: Optimizer confidence score

### Advanced Metrics
- **Junk Kept Rate**: Percentage of junk items retained
- **Intent Preservation**: Primary domain intent maintained
- **Precision/Recall/F1**: Against gold standard labels
- **Calibration Error**: Confidence vs. actual performance alignment

## Acceptance Gates

### Entropy Optimizer
- Median reduction ≥ 60%
- Mean retention ≥ 0.92
- Junk kept rate ≤ 10%

### Pattern Optimizer
- Median reduction ≥ 20%
- Mean retention ≥ 0.90
- Junk kept rate ≤ 15%

### General
- Intent preservation ≥ 95%
- Entropy latency ≤ 1.25× Pattern latency

## Output Files

Results are saved to `output/runs/{timestamp}/`:

- `report.html` - Interactive HTML report
- `results.jsonl` - Detailed per-test results
- `summary.csv` - Aggregated statistics
- `calibration.csv` - Confidence calibration analysis
- `suggested_tweaks.yaml` - Parameter recommendations (if gates fail)

## API Requirements

Your CEGO endpoints should accept:

```json
{
  "query": "search query string",
  "items": ["item 1 text", "item 2 text", ...]
}
```

And return:

```json
{
  "optimized_context": ["kept item 1", "kept item 2"],
  "token_reduction_percentage": 65.5,
  "semantic_retention": 0.92,
  "confidence": 0.88
}
```

## License

MIT License - See LICENSE file for details.