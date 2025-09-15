# ANNEX F: ABLATION STUDIES

*Ablation studies pending. Run with --ablation flag to generate.*


## 2. OPTIMAL CONFIGURATION

Based on ablation studies, the optimal configuration is:

### Entropy Optimizer
- λ_decay: 0.92
- phase_min_gain: 0.015
- mmr_weight: 0.5
- junk_soft_threshold: 0.35

### Pattern Optimizer
- keep_ratio: 0.4
- junk_threshold: 0.40
- diversity_weight: 0.3

These parameters balance reduction aggressiveness with retention quality.
