"""Core metrics calculation for CEGO benchmark results."""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import difflib

from .loaders import TestCase, count_tokens
from .adapters import OptimizationResult, OptimizerType


@dataclass
class BenchmarkMetrics:
    """Complete metrics for a single benchmark run."""
    test_id: str
    optimizer: OptimizerType
    success: bool

    # Core metrics
    input_tokens: int
    output_tokens: int
    token_reduction_percentage: Optional[float]
    semantic_retention: Optional[float]
    confidence: Optional[float]
    latency_ms: float

    # Derived metrics
    kept_indices: List[int]
    junk_kept_rate: Optional[float]
    intent_preservation: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]

    # Error info
    error_message: Optional[str] = None


def calculate_kept_indices(original_items: List[str],
                          optimized_context: List[str],
                          similarity_threshold: float = 0.72) -> List[int]:
    """Map optimized context back to original item indices.

    Uses fuzzy matching to handle potential text modifications.

    Args:
        original_items: Original item texts
        optimized_context: Optimized context strings
        similarity_threshold: Minimum similarity for match

    Returns:
        List of indices into original_items for kept items
    """
    kept_indices = []

    for opt_text in optimized_context:
        best_match_idx = -1
        best_similarity = 0.0

        for i, orig_text in enumerate(original_items):
            # Use difflib for fuzzy string matching
            similarity = difflib.SequenceMatcher(
                None, orig_text.lower().strip(), opt_text.lower().strip()
            ).ratio()

            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match_idx = i

        if best_match_idx >= 0 and best_match_idx not in kept_indices:
            kept_indices.append(best_match_idx)

    return sorted(kept_indices)


def calculate_junk_kept_rate(test_case: TestCase, kept_indices: List[int]) -> Optional[float]:
    """Calculate percentage of junk items that were kept.

    Args:
        test_case: Original test case with ground truth labels
        kept_indices: Indices of items that were kept

    Returns:
        Junk kept rate (0.0 to 1.0), or None if no junk labels available
    """
    junk_items = [i for i, item in enumerate(test_case.items)
                  if item.is_junk_gt is True]

    if not junk_items:
        return None

    kept_junk = sum(1 for idx in kept_indices if idx in junk_items)
    return kept_junk / len(junk_items)


def calculate_intent_preservation(test_case: TestCase, kept_indices: List[int]) -> Optional[float]:
    """Calculate intent preservation based on domain distribution.

    Args:
        test_case: Original test case
        kept_indices: Indices of items that were kept

    Returns:
        Intent preservation score (0.0 to 1.0), or None if insufficient data
    """
    if not test_case.gold or 'intent_tags' not in test_case.gold:
        return None

    target_intents = set(test_case.gold['intent_tags'])
    if not target_intents:
        return None

    # Count domain hints in original vs kept items
    original_domains = [item.domain_hint for item in test_case.items if item.domain_hint]
    kept_domains = [test_case.items[i].domain_hint for i in kept_indices
                   if i < len(test_case.items) and test_case.items[i].domain_hint]

    if not original_domains:
        return None

    # Check if primary intent domains are preserved
    domain_counts_orig = defaultdict(int)
    domain_counts_kept = defaultdict(int)

    for domain in original_domains:
        domain_counts_orig[domain] += 1

    for domain in kept_domains:
        domain_counts_kept[domain] += 1

    # Primary domain should be preserved
    if not domain_counts_orig:
        return 1.0

    primary_domain = max(domain_counts_orig.keys(), key=domain_counts_orig.get)

    return 1.0 if primary_domain in domain_counts_kept else 0.0


def calculate_precision_recall_f1(test_case: TestCase, kept_indices: List[int]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculate precision, recall, F1 against gold standard.

    Args:
        test_case: Test case with gold labels
        kept_indices: Indices of items that were kept

    Returns:
        Tuple of (precision, recall, f1), each Optional[float]
    """
    if not test_case.gold or 'kept_item_ids' not in test_case.gold:
        return None, None, None

    gold_ids = set(test_case.gold['kept_item_ids'])
    if not gold_ids:
        return None, None, None

    # Map kept indices to item IDs
    kept_ids = set()
    for idx in kept_indices:
        if idx < len(test_case.items):
            kept_ids.add(test_case.items[idx].id)

    if not kept_ids:
        # All rejected - precision undefined, recall = 0
        return None, 0.0, 0.0

    true_positives = len(gold_ids.intersection(kept_ids))
    false_positives = len(kept_ids - gold_ids)
    false_negatives = len(gold_ids - kept_ids)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def calculate_benchmark_metrics(test_case: TestCase,
                              result: OptimizationResult) -> BenchmarkMetrics:
    """Calculate complete metrics for a benchmark result.

    Args:
        test_case: Original test case
        result: Optimization result from API

    Returns:
        Complete benchmark metrics
    """
    # Calculate input/output tokens
    input_tokens = count_tokens(test_case.query)
    for item in test_case.items:
        input_tokens += count_tokens(item.text)

    output_tokens = 0
    kept_indices = []
    token_reduction_percentage = result.token_reduction_percentage

    if result.success and result.optimized_context:
        for text in result.optimized_context:
            output_tokens += count_tokens(text)

        # Calculate token reduction percentage if not provided by API
        if token_reduction_percentage is None and input_tokens > 0:
            token_reduction_percentage = 1.0 - (output_tokens / input_tokens)

        # Calculate kept indices using fuzzy matching (API doesn't provide this)
        original_texts = [item.text for item in test_case.items]
        kept_indices = calculate_kept_indices(original_texts, result.optimized_context)

    # Calculate additional metrics
    junk_kept_rate = calculate_junk_kept_rate(test_case, kept_indices) if result.success else None
    intent_preservation = calculate_intent_preservation(test_case, kept_indices) if result.success else None
    precision, recall, f1_score = calculate_precision_recall_f1(test_case, kept_indices) if result.success else (None, None, None)

    return BenchmarkMetrics(
        test_id=test_case.id,
        optimizer=result.optimizer,
        success=result.success,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_reduction_percentage=token_reduction_percentage,
        semantic_retention=result.semantic_retention,
        confidence=result.confidence,  # Will be None since API doesn't provide
        latency_ms=result.latency_ms,
        kept_indices=kept_indices,
        junk_kept_rate=junk_kept_rate,
        intent_preservation=intent_preservation,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        error_message=result.error_message
    )


def calculate_divergence_metrics(pattern_metrics: BenchmarkMetrics,
                               entropy_metrics: BenchmarkMetrics) -> Dict[str, float]:
    """Calculate divergence between pattern and entropy results.

    Args:
        pattern_metrics: Pattern optimizer metrics
        entropy_metrics: Entropy optimizer metrics

    Returns:
        Dictionary with divergence metrics
    """
    if not (pattern_metrics.success and entropy_metrics.success):
        return {"jaccard_divergence": 1.0, "retention_delta": 0.0}

    # Jaccard divergence on kept indices
    pattern_kept = set(pattern_metrics.kept_indices)
    entropy_kept = set(entropy_metrics.kept_indices)

    if not pattern_kept and not entropy_kept:
        jaccard_similarity = 1.0
    elif not pattern_kept or not entropy_kept:
        jaccard_similarity = 0.0
    else:
        intersection = len(pattern_kept.intersection(entropy_kept))
        union = len(pattern_kept.union(entropy_kept))
        jaccard_similarity = intersection / union

    jaccard_divergence = 1.0 - jaccard_similarity

    # Retention delta
    pattern_retention = pattern_metrics.semantic_retention or 0.0
    entropy_retention = entropy_metrics.semantic_retention or 0.0
    retention_delta = abs(pattern_retention - entropy_retention)

    return {
        "jaccard_divergence": jaccard_divergence,
        "retention_delta": retention_delta
    }


def calculate_expected_calibration_error(metrics_list: List[BenchmarkMetrics],
                                       num_bins: int = 10) -> float:
    """Calculate Expected Calibration Error for confidence scores.

    Args:
        metrics_list: List of metrics with confidence and retention scores
        num_bins: Number of confidence bins

    Returns:
        Expected Calibration Error (0.0 to 1.0)
    """
    # Filter successful results with both confidence and retention
    valid_metrics = [
        m for m in metrics_list
        if m.success and m.confidence is not None and m.semantic_retention is not None
    ]

    if len(valid_metrics) < 2:
        return 0.0

    # Create confidence bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(valid_metrics)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find metrics in this confidence bin
        in_bin = [
            m for m in valid_metrics
            if bin_lower <= m.confidence < bin_upper or
               (bin_upper == 1.0 and m.confidence == 1.0)  # Include 1.0 in last bin
        ]

        if not in_bin:
            continue

        # Calculate average confidence and accuracy for this bin
        avg_confidence = np.mean([m.confidence for m in in_bin])
        avg_accuracy = np.mean([m.semantic_retention for m in in_bin])

        # Add weighted contribution to ECE
        bin_weight = len(in_bin) / total_samples
        ece += bin_weight * abs(avg_confidence - avg_accuracy)

    return ece