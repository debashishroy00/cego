
# ANNEX B: METHODOLOGY

## 1. EXPERIMENTAL DESIGN

### 1.1 Benchmark Configuration
- **Datasets**: 5 domains
- **Test Cases**: 50 total
- **Repetitions**: 3
- **Max Items**: 50

### 1.2 Domains Tested
- **dupes**: 10 test cases
- **format**: 10 test cases
- **insurance**: 10 test cases
- **mixed**: 10 test cases
- **sdlc**: 10 test cases


## 2. EVALUATION METRICS

### 2.1 Core Metrics
- **Token Reduction Percentage**: (1 - output_tokens/input_tokens) × 100
- **Semantic Retention**: Cosine similarity between original and optimized embeddings
- **Latency**: End-to-end response time in milliseconds
- **Confidence Score**: Model confidence in optimization quality

### 2.2 Advanced Metrics
- **Junk Kept Rate**: Proportion of identified junk items retained
- **Intent Preservation**: Maintenance of primary query intent
- **Precision/Recall/F1**: Against gold standard labels
- **Expected Calibration Error**: Confidence vs actual performance alignment

## 3. SCORING METHODS

### 3.1 Similarity Scoring Pipeline
1. **Embedding-based**: Sentence transformers with cosine similarity
2. **TF-IDF Fallback**: Scikit-learn TF-IDF for non-embedding scenarios
3. **ROUGE Metrics**: Lexical overlap measurement

### 3.2 Aggregation Strategy
- Median for reduction percentages (robust to outliers)
- Mean for retention scores (captures overall performance)
- P90/P95 for latency (tail performance matters)

## 4. ACCEPTANCE CRITERIA

### 4.1 Entropy Optimizer
- Median reduction ≥ 60%
- Mean retention ≥ 0.92
- Junk kept rate ≤ 10%
- Intent preservation ≥ 95%

### 4.2 Pattern Optimizer
- Median reduction ≥ 20%
- Mean retention ≥ 0.90
- Junk kept rate ≤ 15%
- Intent preservation ≥ 95%

### 4.3 Comparative
- Entropy latency P90 ≤ 1.25× Pattern latency P90

## 5. STATISTICAL RIGOR

- **Confidence Intervals**: 95% CI for all metrics
- **Multiple Runs**: {config.get('runs', {}).get('repeat', 3)} repetitions per test
- **Domain Stratification**: Separate analysis per domain
- **Outlier Handling**: Median-based aggregation for robustness

## 6. EDGE CASE COVERAGE

- **Noisy Data**: Corrupted/incomplete inputs
- **Duplicates**: Near-duplicate content handling
- **Format Variations**: Different text formatting styles
- **Mixed Intent**: Multi-domain queries
- **Scale Testing**: 3-100 items per test case
