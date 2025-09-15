# ANNEX C: AGGREGATE STATISTICS

## 1. OVERALL PERFORMANCE TABLE

| Domain | Optimizer | Median Reduction | Mean Retention | Junk Kept | Intent Preserved | P50 Latency | P90 Latency | Gate Status |
|--------|-----------|-----------------|----------------|-----------|------------------|-------------|-------------|-------------|
| insurance | PATTERN | 0.0% | 0.000 | 0.0% | 100.0% | 15.3ms | 17.2ms | ❌ |
| insurance | ENTROPY | 0.8% | 0.000 | 0.0% | 100.0% | 20.8ms | 22.8ms | ❌ |
| sdlc | PATTERN | 0.0% | 0.200 | 0.0% | 100.0% | 13.7ms | 16.5ms | ❌ |
| sdlc | ENTROPY | 0.8% | 0.100 | 0.0% | 100.0% | 18.1ms | 20.3ms | ❌ |
| mixed | PATTERN | 0.0% | 0.000 | 0.0% | 100.0% | 14.2ms | 16.7ms | ❌ |
| mixed | ENTROPY | 0.8% | 0.000 | 0.0% | 30.0% | 19.1ms | 21.6ms | ❌ |
| dupes | PATTERN | 0.0% | 0.000 | 50.0% | 100.0% | 14.4ms | 16.9ms | ❌ |
| dupes | ENTROPY | 0.8% | 0.000 | 50.0% | 100.0% | 18.2ms | 20.1ms | ❌ |
| format | PATTERN | 0.0% | 0.100 | 0.0% | 100.0% | 15.1ms | 17.6ms | ❌ |
| format | ENTROPY | 0.8% | 0.100 | 0.0% | 100.0% | 19.5ms | 22.0ms | ❌ |

## 2. PERFORMANCE VISUALIZATIONS

### 2.1 Token Reduction Distribution
![Reduction Boxplot](chart_reduction_boxplot.png)

### 2.2 Semantic Retention by Domain
![Retention Bar Chart](chart_retention_barchart.png)

### 2.3 Junk Handling vs Intent Preservation
![Junk Intent Chart](chart_junk_intent.png)

