"""
Phase Transition Detector for Entropy Optimization.
Detects when optimization should stop based on entropy patterns.
"""
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class PhaseTransitionDetector:
    """Detects when optimization should stop based on entropy patterns."""

    def __init__(self, window_size: int = 5, plateau_threshold: float = 0.001):
        """
        Initialize phase transition detector.

        Args:
            window_size: Number of recent entropy values to analyze
            plateau_threshold: Variance threshold for detecting plateaus
        """
        self.window_size = window_size
        self.plateau_threshold = plateau_threshold
        self.entropy_history = []
        self.gradient_history = []

    def should_stop(self, current_entropy: float, iteration: int) -> Tuple[bool, str]:
        """
        Determine if we've hit a phase transition (optimization barrier).

        Args:
            current_entropy: Current entropy value
            iteration: Current iteration number

        Returns:
            Tuple of (should_stop: bool, reason: str)
        """
        self.entropy_history.append(current_entropy)

        # Calculate gradient (rate of change)
        if len(self.entropy_history) > 1:
            gradient = current_entropy - self.entropy_history[-2]
            self.gradient_history.append(gradient)

        # Need enough history for analysis
        if len(self.entropy_history) < self.window_size:
            return False, "Building entropy history"

        # Check for entropy plateau (no significant change)
        recent_entropies = self.entropy_history[-self.window_size:]
        entropy_variance = np.var(recent_entropies)

        if entropy_variance < self.plateau_threshold:
            logger.debug(f"Entropy plateau detected: variance {entropy_variance:.6f}")
            return True, f"Entropy plateau reached (variance: {entropy_variance:.6f})"

        # Check for diminishing returns
        if len(self.entropy_history) >= self.window_size * 2:
            recent_improvement = recent_entropies[-1] - recent_entropies[0]
            previous_window = self.entropy_history[-self.window_size*2:-self.window_size]
            previous_improvement = previous_window[-1] - previous_window[0]

            if previous_improvement != 0:  # Avoid division by zero
                improvement_ratio = abs(recent_improvement) / abs(previous_improvement)
                if improvement_ratio < 0.1:  # Less than 10% of previous improvement
                    logger.debug(f"Diminishing returns: ratio {improvement_ratio:.3f}")
                    return True, f"Diminishing returns (ratio: {improvement_ratio:.3f})"

        # Check for oscillation (entropy bouncing up and down)
        if len(self.gradient_history) >= 4:
            recent_gradients = self.gradient_history[-4:]
            sign_changes = sum(1 for i in range(1, len(recent_gradients))
                             if np.sign(recent_gradients[i]) != np.sign(recent_gradients[i-1]))

            if sign_changes >= 3:  # More than 3 sign changes in 4 gradients
                logger.debug(f"Oscillation detected: {sign_changes} sign changes")
                return True, f"Oscillation detected ({sign_changes} sign changes)"

        # Check for negative entropy trend (getting worse)
        if len(self.gradient_history) >= 3:
            recent_gradients = self.gradient_history[-3:]
            if all(g < -0.01 for g in recent_gradients):  # Consistently decreasing
                logger.debug("Consistent entropy decrease detected")
                return True, "Entropy consistently decreasing"

        # Check iteration limit
        if iteration >= 50:
            logger.debug(f"Iteration limit reached: {iteration}")
            return True, f"Iteration limit reached ({iteration})"

        # Check for convergence (very small improvements)
        if len(self.gradient_history) >= 3:
            recent_gradients = self.gradient_history[-3:]
            if all(abs(g) < 0.001 for g in recent_gradients):
                logger.debug("Convergence detected: gradients near zero")
                return True, "Convergence detected (minimal gradients)"

        return False, "Continue optimizing"

    def get_statistics(self) -> dict:
        """Get statistics about the optimization process."""
        if not self.entropy_history:
            return {}

        stats = {
            "total_iterations": len(self.entropy_history),
            "initial_entropy": self.entropy_history[0],
            "final_entropy": self.entropy_history[-1],
            "entropy_change": self.entropy_history[-1] - self.entropy_history[0],
            "max_entropy": max(self.entropy_history),
            "min_entropy": min(self.entropy_history),
            "entropy_variance": float(np.var(self.entropy_history))
        }

        if self.gradient_history:
            stats.update({
                "average_gradient": float(np.mean(self.gradient_history)),
                "gradient_variance": float(np.var(self.gradient_history)),
                "max_gradient": float(max(self.gradient_history)),
                "min_gradient": float(min(self.gradient_history))
            })

        return stats

    def reset(self):
        """Reset the detector for a new optimization run."""
        self.entropy_history.clear()
        self.gradient_history.clear()