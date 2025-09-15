# ANNEX C: AGGREGATE STATISTICS

## 1. OVERALL PERFORMANCE TABLE

| Domain | Optimizer | Median Reduction | Mean Retention | Junk Kept | Intent Preserved | P50 Latency | P90 Latency | Gate Status |
|--------|-----------|-----------------|----------------|-----------|------------------|-------------|-------------|-------------|
| insurance | PATTERN | 0.0% | 0.000 | 0.0% | 100.0% | 15.1ms | 16.8ms | ❌ |
| insurance | ENTROPY | 0.8% | 0.000 | 0.0% | 100.0% | 18.4ms | 20.6ms | ❌ |
| sdlc | PATTERN | 0.0% | 0.200 | 0.0% | 100.0% | 14.9ms | 17.3ms | ❌ |
| sdlc | ENTROPY | 0.8% | 0.100 | 0.0% | 100.0% | 18.4ms | 21.6ms | ❌ |
| mixed | PATTERN | 0.0% | 0.000 | 0.0% | 100.0% | 14.0ms | 16.0ms | ❌ |
| mixed | ENTROPY | 0.8% | 0.000 | 0.0% | 30.0% | 18.6ms | 20.6ms | ❌ |
| dupes | PATTERN | 0.0% | 0.000 | 50.0% | 100.0% | 15.3ms | 17.4ms | ❌ |
| dupes | ENTROPY | 0.8% | 0.000 | 50.0% | 100.0% | 18.7ms | 22.2ms | ❌ |
| format | PATTERN | 0.0% | 0.100 | 0.0% | 100.0% | 14.6ms | 16.5ms | ❌ |
| format | ENTROPY | 0.8% | 0.100 | 0.0% | 100.0% | 18.7ms | 20.5ms | ❌ |

## 2. PERFORMANCE VISUALIZATIONS

### 2.1 Token Reduction Distribution
![Reduction Boxplot](chart_reduction_boxplot.png)

### 2.2 Semantic Retention by Domain
![Retention Bar Chart](chart_retention_barchart.png)

### 2.3 Junk Handling vs Intent Preservation
![Junk Intent Chart](chart_junk_intent.png)

