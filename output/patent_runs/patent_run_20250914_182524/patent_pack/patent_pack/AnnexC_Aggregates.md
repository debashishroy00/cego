# ANNEX C: AGGREGATE STATISTICS

## 1. OVERALL PERFORMANCE TABLE

| Domain | Optimizer | Median Reduction | Mean Retention | Junk Kept | Intent Preserved | P50 Latency | P90 Latency | Gate Status |
|--------|-----------|-----------------|----------------|-----------|------------------|-------------|-------------|-------------|
| insurance | PATTERN | 0.0% | 0.000 | 0.0% | 100.0% | 15.9ms | 18.6ms | ❌ |
| insurance | ENTROPY | 0.8% | 0.000 | 0.0% | 100.0% | 22.7ms | 25.1ms | ❌ |
| sdlc | PATTERN | 0.0% | 0.200 | 0.0% | 100.0% | 17.6ms | 19.9ms | ❌ |
| sdlc | ENTROPY | 0.8% | 0.100 | 0.0% | 100.0% | 22.4ms | 25.1ms | ❌ |
| mixed | PATTERN | 0.0% | 0.000 | 0.0% | 100.0% | 16.5ms | 18.9ms | ❌ |
| mixed | ENTROPY | 0.8% | 0.000 | 0.0% | 30.0% | 23.0ms | 25.0ms | ❌ |
| dupes | PATTERN | 0.0% | 0.000 | 50.0% | 100.0% | 16.4ms | 19.0ms | ❌ |
| dupes | ENTROPY | 0.8% | 0.000 | 50.0% | 100.0% | 22.4ms | 24.3ms | ❌ |
| format | PATTERN | 0.0% | 0.100 | 0.0% | 100.0% | 17.7ms | 19.1ms | ❌ |
| format | ENTROPY | 0.8% | 0.100 | 0.0% | 100.0% | 22.4ms | 24.1ms | ❌ |

## 2. PERFORMANCE VISUALIZATIONS

### 2.1 Token Reduction Distribution
![Reduction Boxplot](chart_reduction_boxplot.png)

### 2.2 Semantic Retention by Domain
![Retention Bar Chart](chart_retention_barchart.png)

### 2.3 Junk Handling vs Intent Preservation
![Junk Intent Chart](chart_junk_intent.png)

