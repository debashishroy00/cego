"""
Enhanced confidence calibrator with expanded range and performance-based adjustments.
Addresses narrow confidence distribution and poor calibration issues.
"""
import numpy as np
from typing import Optional, List, Tuple


class ConfidenceCalibrator:
    """Enhanced confidence calibrator with temperature scaling and performance signals."""

    def __init__(self):
        # Current raw confidence range from API
        self.raw_min = 0.14
        self.raw_max = 0.20

        # Target calibrated range
        self.target_min = 0.15
        self.target_max = 0.75

        # Temperature for distribution expansion
        self.temperature = 1.8

    def calibrate_confidence(self,
                           raw_confidence: Optional[float],
                           reduction: Optional[float] = None,
                           diversity: Optional[float] = None) -> Optional[float]:
        """
        Calibrate confidence from narrow 14-20% range to usable 15-75% range.

        Args:
            raw_confidence: Raw confidence from entropy analysis (typically 0.14-0.20)
            reduction: Token reduction percentage (0.0-1.0) for performance adjustment
            diversity: Entropy diversity score (0.0-1.0) for stability adjustment

        Returns:
            Calibrated confidence in [0.15, 0.75] range, or None if input is None
        """
        if raw_confidence is None:
            return None

        # Normalize raw confidence to [0,1] with clipping
        raw_clipped = float(np.clip(raw_confidence, self.raw_min, self.raw_max))
        normalized = (raw_clipped - self.raw_min) / (self.raw_max - self.raw_min)
        normalized = float(np.clip(normalized, 0.0, 1.0))

        # Apply temperature scaling to expand distribution
        # Temperature < 1 would compress, > 1 expands high-confidence region
        expanded = normalized ** (1 / self.temperature)

        # Map to target range [0.15, 0.75]
        conf = self.target_min + (self.target_max - self.target_min) * expanded

        # Performance-based adjustments
        if reduction is not None:
            if reduction > 0.80:  # High performance boost
                conf += 0.05
            elif reduction > 0.75:  # Good performance
                conf += 0.03
            elif reduction < 0.50:  # Lower performance penalty
                conf -= 0.05
            elif reduction < 0.60:  # Moderate performance penalty
                conf -= 0.03

        # Diversity-based stability adjustment
        # Low diversity (< 0.5) suggests overconfidence, penalize slightly
        if diversity is not None:
            diversity_penalty = 0.05 * max(0.0, 0.5 - diversity)
            conf -= diversity_penalty

        # Ensure final bounds
        return float(np.clip(conf, self.target_min, self.target_max))

    def calculate_calibration_bins(self,
                                 confidence_scores: List[float],
                                 accuracy_scores: List[float],
                                 n_bins: int = 10) -> Tuple[float, List[dict]]:
        """
        Calculate Expected Calibration Error (ECE) across confidence bins.

        Args:
            confidence_scores: List of confidence predictions
            accuracy_scores: List of actual accuracy/performance scores
            n_bins: Number of bins for calibration analysis

        Returns:
            Tuple of (ECE, bin_details) where bin_details is list of dicts with bin info
        """
        if not confidence_scores or not accuracy_scores:
            return 0.0, []

        if len(confidence_scores) != len(accuracy_scores):
            raise ValueError("Confidence and accuracy lists must have same length")

        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        total_samples = len(confidence_scores)
        bin_details = []

        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            # Find samples in this bin
            in_bin_indices = []
            for j, conf in enumerate(confidence_scores):
                if bin_lower <= conf < bin_upper or (bin_upper == 1.0 and conf == 1.0):
                    in_bin_indices.append(j)

            if not in_bin_indices:
                bin_details.append({
                    "bin_id": i,
                    "bin_lower": bin_lower,
                    "bin_upper": bin_upper,
                    "count": 0,
                    "avg_confidence": None,
                    "avg_accuracy": None,
                    "calibration_error": 0.0
                })
                continue

            # Calculate bin statistics
            bin_confidences = [confidence_scores[j] for j in in_bin_indices]
            bin_accuracies = [accuracy_scores[j] for j in in_bin_indices]

            avg_confidence = np.mean(bin_confidences)
            avg_accuracy = np.mean(bin_accuracies)
            calibration_error = abs(avg_confidence - avg_accuracy)

            # Weight by bin size for ECE
            bin_weight = len(in_bin_indices) / total_samples
            ece += bin_weight * calibration_error

            bin_details.append({
                "bin_id": i,
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "count": len(in_bin_indices),
                "avg_confidence": avg_confidence,
                "avg_accuracy": avg_accuracy,
                "calibration_error": calibration_error,
                "weight": bin_weight
            })

        return ece, bin_details

    def get_calibration_stats(self, confidence_scores: List[float]) -> dict:
        """Get basic statistics about confidence distribution."""
        if not confidence_scores:
            return {}

        conf_array = np.array(confidence_scores)

        return {
            "count": len(confidence_scores),
            "mean": float(np.mean(conf_array)),
            "median": float(np.median(conf_array)),
            "std": float(np.std(conf_array)),
            "min": float(np.min(conf_array)),
            "max": float(np.max(conf_array)),
            "q25": float(np.percentile(conf_array, 25)),
            "q75": float(np.percentile(conf_array, 75)),
            "range": float(np.max(conf_array) - np.min(conf_array)),
            "unique_values": len(np.unique(np.round(conf_array, 2)))
        }

    def validate_confidence_distribution(self, confidence_scores: List[float]) -> dict:
        """Validate that confidence distribution is reasonable."""
        stats = self.get_calibration_stats(confidence_scores)

        issues = []

        # Check for narrow range (should use > 50% of target range)
        expected_range = self.target_max - self.target_min
        if stats.get("range", 0) < 0.5 * expected_range:
            issues.append(f"Narrow range: {stats['range']:.3f} < {0.5 * expected_range:.3f}")

        # Check for sufficient diversity (should have > 5 unique values per 20 samples)
        min_unique = max(3, len(confidence_scores) // 4)
        if stats.get("unique_values", 0) < min_unique:
            issues.append(f"Low diversity: {stats['unique_values']} < {min_unique}")

        # Check for reasonable mean (should be in central 60% of range)
        central_min = self.target_min + 0.2 * expected_range
        central_max = self.target_max - 0.2 * expected_range
        if not (central_min <= stats.get("mean", 0) <= central_max):
            issues.append(f"Mean outside central range: {stats['mean']:.3f} not in [{central_min:.3f}, {central_max:.3f}]")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "stats": stats
        }


def test_confidence_calibrator():
    """Test confidence calibrator functionality."""
    calibrator = ConfidenceCalibrator()

    # Test basic calibration
    raw_conf = 0.175  # Typical raw confidence
    calibrated = calibrator.calibrate_confidence(raw_conf, reduction=0.80, diversity=0.6)

    assert 0.15 <= calibrated <= 0.75, f"Calibrated confidence {calibrated} out of range"
    assert calibrated > 0.25, f"High performance should boost confidence above 0.25, got {calibrated}"

    print(f"[OK] Raw confidence {raw_conf:.3f} -> {calibrated:.3f}")

    # Test edge cases
    assert calibrator.calibrate_confidence(None) is None

    # Test bounds
    low_conf = calibrator.calibrate_confidence(0.14, reduction=0.30)
    high_conf = calibrator.calibrate_confidence(0.20, reduction=0.85)

    assert low_conf < high_conf, f"Higher reduction should yield higher confidence"
    assert 0.15 <= low_conf <= 0.75, f"Low confidence {low_conf} out of bounds"
    assert 0.15 <= high_conf <= 0.75, f"High confidence {high_conf} out of bounds"

    print("[OK] Confidence calibrator tests passed")


def test_calibration_bins():
    """Test ECE calculation."""
    calibrator = ConfidenceCalibrator()

    # Perfect calibration
    perfect_conf = [0.2, 0.4, 0.6, 0.8]
    perfect_acc = [0.2, 0.4, 0.6, 0.8]

    ece, bins = calibrator.calculate_calibration_bins(perfect_conf, perfect_acc, n_bins=4)
    assert ece < 0.05, f"Perfect calibration should have low ECE, got {ece:.3f}"

    # Poor calibration
    poor_conf = [0.8, 0.8, 0.8, 0.8]
    poor_acc = [0.2, 0.3, 0.4, 0.5]

    ece_poor, _ = calibrator.calculate_calibration_bins(poor_conf, poor_acc, n_bins=4)
    assert ece_poor > 0.3, f"Poor calibration should have high ECE, got {ece_poor:.3f}"

    print(f"[OK] ECE calculation: perfect={ece:.3f}, poor={ece_poor:.3f}")


if __name__ == "__main__":
    test_confidence_calibrator()
    test_calibration_bins()

    # Demo expanded confidence range
    calibrator = ConfidenceCalibrator()

    print("\nConfidence expansion demo:")
    test_cases = [
        (0.14, 0.50, 0.3),  # Low raw, medium performance, low diversity
        (0.175, 0.75, 0.7), # Medium raw, good performance, good diversity
        (0.20, 0.85, 0.9),  # High raw, excellent performance, high diversity
    ]

    for raw, reduction, diversity in test_cases:
        calibrated = calibrator.calibrate_confidence(raw, reduction, diversity)
        print(f"  Raw: {raw:.3f}, Red: {reduction:.2f}, Div: {diversity:.1f} -> Calibrated: {calibrated:.3f}")