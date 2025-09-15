"""
Robust sanity checks for benchmark results to prevent uniform distributions
and catch data integrity issues before report generation.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SanityCheckConfig:
    """Configuration for sanity check bounds and tolerances."""
    # Tolerance for reduction calculation verification
    reduction_tolerance: float = 0.02  # Relative tolerance

    # Minimum unique values required
    min_unique_reductions: int = 4
    min_unique_retentions: int = 4

    # Variance bounds [min, max] - adjusted for realistic data
    reduction_variance_bounds: Tuple[float, float] = (0.02, 0.30)
    retention_variance_bounds: Tuple[float, float] = (0.03, 0.40)

    # Performance bounds [min, max]
    reduction_bounds: Tuple[float, float] = (0.20, 0.95)
    retention_bounds: Tuple[float, float] = (0.10, 0.98)

    # Confidence bounds [min, max]
    confidence_bounds: Tuple[float, float] = (0.10, 0.90)

    # ECE bounds
    max_ece: float = 0.50


class MetricsSanityChecker:
    """Comprehensive sanity checker for benchmark metrics."""

    def __init__(self, config: Optional[SanityCheckConfig] = None):
        self.config = config or SanityCheckConfig()
        self.issues = []

    def check_individual_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate individual optimization result for basic integrity.

        Args:
            result: Single optimization result dictionary

        Returns:
            True if result passes checks, raises AssertionError otherwise
        """
        # Check reduction calculation consistency
        if (result.get('token_reduction_percentage') is not None and
            result.get('input_tokens', 0) > 0):

            in_tokens = result['input_tokens']
            out_tokens = result['output_tokens']
            actual_reduction = result['token_reduction_percentage']

            expected_reduction = 1.0 - (out_tokens / in_tokens)

            # Use relative tolerance to handle small inputs gracefully
            relative_error = abs(actual_reduction - expected_reduction) / max(expected_reduction, 0.01)

            assert relative_error <= self.config.reduction_tolerance, \
                f"Reduction calculation mismatch: actual={actual_reduction:.3f}, " \
                f"expected={expected_reduction:.3f}, error={relative_error:.3f}"

        # Check semantic retention bounds
        retention = result.get('semantic_retention')
        if retention is not None:
            assert 0.0 <= retention <= 1.0, \
                f"Retention {retention:.3f} outside valid range [0.0, 1.0]"

        # Check confidence bounds (if present)
        confidence = result.get('confidence')
        if confidence is not None:
            min_conf, max_conf = self.config.confidence_bounds
            assert min_conf <= confidence <= max_conf, \
                f"Confidence {confidence:.3f} outside expected range [{min_conf:.2f}, {max_conf:.2f}]"

        # Check latency reasonableness
        latency = result.get('latency_ms')
        if latency is not None:
            assert 0.0 < latency < 10000.0, \
                f"Latency {latency:.1f}ms outside reasonable range (0, 10000ms)"

        return True

    def check_aggregate_distribution(self,
                                   results: List[Dict[str, Any]],
                                   optimizer_name: str = "unknown") -> bool:
        """
        Validate aggregate distributions for suspicious patterns.

        Args:
            results: List of optimization results for one optimizer
            optimizer_name: Name of optimizer for error messages

        Returns:
            True if distributions pass checks, raises AssertionError otherwise
        """
        if len(results) < 3:
            # Skip checks for very small samples
            return True

        # Extract metrics
        reductions = self._extract_metric(results, 'token_reduction_percentage')
        retentions = self._extract_metric(results, 'semantic_retention')
        latencies = self._extract_metric(results, 'latency_ms')
        confidences = self._extract_metric(results, 'confidence')

        # Check diversity (anti-uniformity)
        if reductions:
            # Adjust min_unique based on sample size
            min_unique_reductions = min(self.config.min_unique_reductions, len(reductions) // 3 + 1)
            self._check_diversity(reductions, f"{optimizer_name} reduction", precision=2,
                                min_unique=min_unique_reductions)

        if retentions:
            min_unique_retentions = min(self.config.min_unique_retentions, len(retentions) // 3 + 1)
            self._check_diversity(retentions, f"{optimizer_name} retention", precision=2,
                                min_unique=min_unique_retentions)

        # Check variance bounds
        if reductions:
            self._check_variance_bounds(reductions, f"{optimizer_name} reduction",
                                      self.config.reduction_variance_bounds)

        if retentions:
            self._check_variance_bounds(retentions, f"{optimizer_name} retention",
                                      self.config.retention_variance_bounds)

        # Check performance bounds
        if reductions:
            self._check_performance_bounds(reductions, f"{optimizer_name} reduction",
                                         self.config.reduction_bounds)

        if retentions:
            self._check_performance_bounds(retentions, f"{optimizer_name} retention",
                                         self.config.retention_bounds)

        # Check latency distribution
        if latencies:
            self._check_latency_distribution(latencies, optimizer_name)

        # Check confidence distribution (if available)
        if confidences:
            self._check_confidence_distribution(confidences, optimizer_name)

        return True

    def check_comparative_results(self,
                                pattern_results: List[Dict[str, Any]],
                                entropy_results: List[Dict[str, Any]]) -> bool:
        """
        Validate comparative performance between optimizers.

        Args:
            pattern_results: Results from pattern optimizer
            entropy_results: Results from entropy optimizer

        Returns:
            True if comparison passes checks, raises AssertionError otherwise
        """
        if not pattern_results or not entropy_results:
            return True

        pattern_reductions = self._extract_metric(pattern_results, 'token_reduction_percentage')
        entropy_reductions = self._extract_metric(entropy_results, 'token_reduction_percentage')

        if pattern_reductions and entropy_reductions:
            pattern_median = np.median(pattern_reductions)
            entropy_median = np.median(entropy_reductions)

            # Entropy should generally outperform pattern (allow 5% tolerance)
            advantage = entropy_median - pattern_median
            min_advantage = -0.05  # Allow slight underperformance

            assert advantage >= min_advantage, \
                f"Entropy median reduction ({entropy_median:.3f}) should generally " \
                f"exceed pattern ({pattern_median:.3f}), got advantage {advantage:.3f}"

            # But advantage shouldn't be impossibly large
            max_advantage = 0.50  # 50 percentage points max
            assert advantage <= max_advantage, \
                f"Entropy advantage {advantage:.3f} seems unrealistically large (max {max_advantage:.3f})"

        return True

    def _extract_metric(self, results: List[Dict[str, Any]], metric_name: str) -> List[float]:
        """Extract non-null metric values from results."""
        values = []
        for result in results:
            value = result.get(metric_name)
            if value is not None:
                values.append(float(value))
        return values

    def _check_diversity(self, values: List[float], metric_name: str,
                        precision: int, min_unique: int):
        """Check for suspicious uniformity in metric values."""
        if not values:
            return

        # Round to specified precision and count unique values
        rounded_values = [round(v, precision) for v in values]
        unique_count = len(set(rounded_values))

        assert unique_count >= min_unique, \
            f"{metric_name} shows only {unique_count} unique values " \
            f"(expected ≥{min_unique}) - likely uniformity bug"

        # Check for excessive clustering around single value
        most_common = max(set(rounded_values), key=rounded_values.count)
        most_common_count = rounded_values.count(most_common)
        clustering_ratio = most_common_count / len(rounded_values)

        # Allow higher clustering for small samples, strict for large samples
        max_clustering = min(0.70, 0.90 - 0.10 * (len(values) / 10))

        assert clustering_ratio <= max_clustering, \
            f"{metric_name} shows {clustering_ratio:.1%} clustering on value {most_common:.3f} " \
            f"(max allowed: {max_clustering:.1%}) - suspicious uniformity"

    def _check_variance_bounds(self, values: List[float], metric_name: str,
                              bounds: Tuple[float, float]):
        """Check that variance is within expected bounds."""
        if len(values) < 3:
            return

        variance = float(np.var(values))
        std_dev = float(np.std(values))
        min_var, max_var = bounds

        assert min_var <= std_dev <= max_var, \
            f"{metric_name} std deviation {std_dev:.3f} outside expected range " \
            f"[{min_var:.3f}, {max_var:.3f}] - may indicate bug or unrealistic data"

    def _check_performance_bounds(self, values: List[float], metric_name: str,
                                 bounds: Tuple[float, float]):
        """Check that performance metrics are within plausible bounds."""
        mean_value = float(np.mean(values))
        min_bound, max_bound = bounds

        assert min_bound <= mean_value <= max_bound, \
            f"{metric_name} mean {mean_value:.3f} outside plausible range " \
            f"[{min_bound:.3f}, {max_bound:.3f}]"

    def _check_latency_distribution(self, latencies: List[float], optimizer_name: str):
        """Check latency distribution for reasonableness."""
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)

        # Check for reasonable mean latency (1ms to 1000ms typical range)
        assert 1.0 <= mean_latency <= 1000.0, \
            f"{optimizer_name} mean latency {mean_latency:.1f}ms outside reasonable range [1, 1000ms]"

        # Check for reasonable variance (shouldn't be too uniform or too scattered)
        cv = std_latency / mean_latency  # Coefficient of variation
        assert 0.05 <= cv <= 0.50, \
            f"{optimizer_name} latency CV {cv:.3f} outside expected range [0.05, 0.50] " \
            f"- may indicate synthetic timing"

    def _check_confidence_distribution(self, confidences: List[float], optimizer_name: str):
        """Check confidence distribution for expanded range and reasonable spread."""
        conf_range = np.max(confidences) - np.min(confidences)
        unique_values = len(set(np.round(confidences, 2)))

        # Should use significant portion of available confidence range
        min_expected_range = 0.20  # Should span at least 20 percentage points
        assert conf_range >= min_expected_range, \
            f"{optimizer_name} confidence range {conf_range:.3f} too narrow " \
            f"(expected ≥{min_expected_range:.3f}) - calibration may not be working"

        # Should have sufficient diversity
        min_unique = max(3, len(confidences) // 10)
        assert unique_values >= min_unique, \
            f"{optimizer_name} confidence has only {unique_values} unique values " \
            f"(expected ≥{min_unique}) - insufficient diversity"


def validate_benchmark_results(all_results: List[Dict[str, Any]],
                              config: Optional[SanityCheckConfig] = None) -> bool:
    """
    Complete validation of benchmark results.

    Args:
        all_results: List of all optimization results
        config: Optional sanity check configuration

    Returns:
        True if all checks pass

    Raises:
        AssertionError: If any sanity check fails
        ValueError: If input data is invalid
    """
    if not all_results:
        raise ValueError("No results provided for validation")

    checker = MetricsSanityChecker(config)

    print(f"[INFO] Validating {len(all_results)} benchmark results...")

    # Individual result checks
    for i, result in enumerate(all_results):
        try:
            checker.check_individual_result(result)
        except AssertionError as e:
            raise AssertionError(f"Individual result {i} failed: {e}")

    # Group by optimizer
    pattern_results = [r for r in all_results if r.get('optimizer') == 'pattern']
    entropy_results = [r for r in all_results if r.get('optimizer') == 'entropy']

    print(f"[INFO] Found {len(pattern_results)} pattern results, {len(entropy_results)} entropy results")

    # Aggregate distribution checks
    if pattern_results:
        checker.check_aggregate_distribution(pattern_results, "pattern")

    if entropy_results:
        checker.check_aggregate_distribution(entropy_results, "entropy")

    # Comparative checks
    if pattern_results and entropy_results:
        checker.check_comparative_results(pattern_results, entropy_results)

    print("[OK] All sanity checks passed")
    return True


def test_sanity_checker():
    """Test sanity checker with various scenarios."""

    # Valid diverse results with realistic variance
    import random
    random.seed(42)  # Reproducible test

    valid_results = []
    for i in range(20):
        # Create realistic diverse reduction percentages
        if i % 2 == 0:  # Pattern optimizer
            reduction = 0.55 + random.uniform(-0.08, 0.08)  # 47-63% range
        else:  # Entropy optimizer
            reduction = 0.75 + random.uniform(-0.05, 0.05)  # 70-80% range

        # Calculate consistent tokens
        in_tokens = 100 + random.randint(0, 50)
        out_tokens = int(in_tokens * (1 - reduction))

        valid_results.append({
            'optimizer': 'pattern' if i % 2 == 0 else 'entropy',
            'input_tokens': in_tokens,
            'output_tokens': out_tokens,
            'token_reduction_percentage': reduction,
            'semantic_retention': 0.65 + random.uniform(0, 0.30),  # More realistic retention spread
            'confidence': 0.2 + random.uniform(0, 0.5) if i % 2 == 1 else None,
            'latency_ms': 15.0 + random.uniform(-5, 10),  # Realistic latency spread
        })

    # Should pass
    validate_benchmark_results(valid_results)
    print("[OK] Valid results passed")

    # Test uniform results (should fail)
    uniform_results = []
    for i in range(10):
        uniform_results.append({
            'optimizer': 'pattern',
            'input_tokens': 100,
            'output_tokens': 50,
            'token_reduction_percentage': 0.5,  # Identical
            'semantic_retention': 0.8,  # Identical
            'latency_ms': 15.0,
        })

    try:
        validate_benchmark_results(uniform_results)
        assert False, "Should have failed on uniform results"
    except AssertionError as e:
        assert "unique values" in str(e).lower()
        print("[OK] Correctly detected uniform results")

    # Test inconsistent reduction calculation (should fail)
    inconsistent_results = [{
        'optimizer': 'pattern',
        'input_tokens': 100,
        'output_tokens': 50,
        'token_reduction_percentage': 0.8,  # Should be 0.5
        'semantic_retention': 0.8,
        'latency_ms': 15.0,
    }]

    try:
        validate_benchmark_results(inconsistent_results)
        assert False, "Should have failed on inconsistent reduction"
    except AssertionError as e:
        assert "mismatch" in str(e).lower()
        print("[OK] Correctly detected reduction calculation error")


if __name__ == "__main__":
    test_sanity_checker()
    print("\n[OK] All sanity checker tests passed")