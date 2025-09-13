"""
Dynamic Lambda (λ) Tuner for CEGO optimization.

Component ID: C-002
Priority: Critical

This module implements dynamic lambda tuning based on the functional specification:
λ = λ_base × domain_factor × performance_factor × exploration_decay

The lambda parameter controls the balance between entropy and relevance in optimization,
adapting to domain characteristics and performance feedback over time.

Mathematical Foundation:
- λ_base: Base lambda values for different optimization modes
- domain_factor: Learned adjustment factor per content domain
- performance_factor: Dynamic adjustment based on recent feedback scores
- exploration_decay: Reduces exploration over time for convergence

Features:
- Adaptive lambda calculation based on optimization context
- Domain-specific learning and adjustment
- Performance feedback integration
- Exploration vs exploitation balance
- Thread-safe state management
"""

import numpy as np
import logging
import threading
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """Optimization modes with different lambda base values."""
    DEBUG = "debug"
    EXPLORE = "explore"
    ANALYZE = "analyze"
    GENERATE = "generate"


@dataclass
class LambdaResult:
    """Result of lambda calculation with metadata."""
    lambda_value: float
    base_lambda: float
    domain_factor: float
    performance_factor: float
    exploration_decay: float
    mode: OptimizationMode
    confidence: float
    calculation_time_ms: float
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class PerformanceFeedback:
    """Feedback data for lambda tuning."""
    timestamp: datetime
    lambda_used: float
    optimization_mode: OptimizationMode
    domain: str
    token_reduction: float
    accuracy_score: float
    user_satisfaction: float
    processing_time_ms: float
    success: bool


class DynamicLambdaTuner:
    """
    Dynamic Lambda (λ) Tuner implementing adaptive lambda calculation.

    The tuner adjusts lambda values based on:
    1. Optimization mode (debug, explore, analyze, generate)
    2. Domain characteristics (learned per content type)
    3. Performance feedback (recent success metrics)
    4. Exploration decay (convergence over time)

    Thread-safe for concurrent usage.
    """

    def __init__(self,
                 feedback_window_size: int = 50,
                 domain_learning_rate: float = 0.1,
                 performance_weight: float = 0.3):
        """
        Initialize the Dynamic Lambda Tuner.

        Args:
            feedback_window_size: Number of recent feedback samples to consider
            domain_learning_rate: Rate of domain factor adaptation (0.0-1.0)
            performance_weight: Weight of performance feedback in calculations
        """
        # Base lambda values per optimization mode (from spec)
        self.base_lambdas = {
            OptimizationMode.DEBUG: 2.0,
            OptimizationMode.EXPLORE: 0.5,
            OptimizationMode.ANALYZE: 1.0,
            OptimizationMode.GENERATE: 1.5
        }

        # Configuration parameters
        self.feedback_window_size = feedback_window_size
        self.domain_learning_rate = domain_learning_rate
        self.performance_weight = performance_weight

        # Domain factors: learned adjustment per domain [0.5, 2.0]
        self.domain_factors = defaultdict(lambda: 1.0)
        self.domain_confidence = defaultdict(lambda: 0.1)

        # Performance feedback history
        self.feedback_history = deque(maxlen=feedback_window_size)

        # Iteration tracking for exploration decay
        self.iteration_counts = defaultdict(int)

        # Thread safety
        self._lock = threading.RLock()

        # Performance cache
        self._performance_cache = {}
        self._cache_expiry = timedelta(minutes=10)

    def calculate_lambda(self,
                        mode: OptimizationMode,
                        domain: str = "general",
                        context_size: Optional[int] = None,
                        query_complexity: Optional[float] = None) -> LambdaResult:
        """
        Calculate dynamic lambda value for optimization.

        Args:
            mode: Optimization mode (debug, explore, analyze, generate)
            domain: Content domain for domain-specific adjustment
            context_size: Size of context pool (influences exploration)
            query_complexity: Complexity score of query (0.0-1.0)

        Returns:
            LambdaResult with calculated lambda and metadata

        Example:
            >>> tuner = DynamicLambdaTuner()
            >>> result = tuner.calculate_lambda(
            ...     OptimizationMode.ANALYZE,
            ...     domain="technical_docs"
            ... )
            >>> print(f"Lambda: {result.lambda_value:.3f}")
        """
        import time
        start_time = time.time()

        with self._lock:
            # Get base lambda for mode
            base_lambda = self.base_lambdas[mode]

            # Get domain factor
            domain_factor = self._get_domain_factor(domain)

            # Calculate performance factor
            performance_factor = self._calculate_performance_factor(domain, mode)

            # Calculate exploration decay
            exploration_decay = self._calculate_exploration_decay(domain, mode)

            # Apply context-specific adjustments
            if context_size is not None:
                exploration_decay *= self._get_context_adjustment(context_size)

            if query_complexity is not None:
                base_lambda *= self._get_complexity_adjustment(query_complexity)

            # Final lambda calculation
            lambda_value = base_lambda * domain_factor * performance_factor * exploration_decay

            # Ensure lambda stays within reasonable bounds
            lambda_value = np.clip(lambda_value, 0.1, 5.0)

            # Calculate confidence based on domain experience and feedback quality
            confidence = self._calculate_confidence(domain, mode)

            # Increment iteration count
            self.iteration_counts[f"{domain}_{mode.value}"] += 1

            processing_time = (time.time() - start_time) * 1000

            return LambdaResult(
                lambda_value=lambda_value,
                base_lambda=base_lambda,
                domain_factor=domain_factor,
                performance_factor=performance_factor,
                exploration_decay=exploration_decay,
                mode=mode,
                confidence=confidence,
                calculation_time_ms=processing_time,
                metadata={
                    'domain': domain,
                    'context_size': context_size,
                    'query_complexity': query_complexity,
                    'iteration_count': self.iteration_counts[f"{domain}_{mode.value}"]
                }
            )

    def provide_feedback(self, feedback: PerformanceFeedback) -> None:
        """
        Provide performance feedback to improve lambda tuning.

        Args:
            feedback: Performance feedback data
        """
        with self._lock:
            # Add to feedback history
            self.feedback_history.append(feedback)

            # Update domain factors based on feedback
            self._update_domain_factor(feedback)

            # Clear performance cache to force recalculation
            self._performance_cache.clear()

            logger.debug(f"Received feedback: domain={feedback.domain}, "
                        f"lambda={feedback.lambda_used:.3f}, "
                        f"success={feedback.success}")

    def get_domain_statistics(self, domain: str = None) -> Dict[str, any]:
        """
        Get statistics for a domain or all domains.

        Args:
            domain: Specific domain or None for all domains

        Returns:
            Dictionary with domain statistics
        """
        with self._lock:
            if domain:
                return {
                    'domain': domain,
                    'factor': self.domain_factors[domain],
                    'confidence': self.domain_confidence[domain],
                    'feedback_count': len([f for f in self.feedback_history if f.domain == domain]),
                    'iteration_count': sum(count for key, count in self.iteration_counts.items()
                                         if key.startswith(f"{domain}_"))
                }
            else:
                return {
                    'domains': dict(self.domain_factors),
                    'confidences': dict(self.domain_confidence),
                    'total_feedback': len(self.feedback_history),
                    'total_iterations': sum(self.iteration_counts.values())
                }

    def reset_domain(self, domain: str) -> None:
        """
        Reset learning for a specific domain.

        Args:
            domain: Domain to reset
        """
        with self._lock:
            self.domain_factors[domain] = 1.0
            self.domain_confidence[domain] = 0.1

            # Reset iteration counts for this domain
            keys_to_reset = [key for key in self.iteration_counts.keys()
                           if key.startswith(f"{domain}_")]
            for key in keys_to_reset:
                self.iteration_counts[key] = 0

            # Clear cache
            self._performance_cache.clear()

            logger.info(f"Reset domain learning for: {domain}")

    def _get_domain_factor(self, domain: str) -> float:
        """Get domain-specific adjustment factor."""
        factor = self.domain_factors[domain]
        # Ensure factor stays within bounds [0.5, 2.0]
        return np.clip(factor, 0.5, 2.0)

    def _calculate_performance_factor(self, domain: str, mode: OptimizationMode) -> float:
        """
        Calculate performance factor based on recent feedback.

        Performance factor = f(recent_feedback_scores)
        Higher scores → factor closer to 1.0
        Lower scores → factor adjusted to improve performance
        """
        cache_key = f"{domain}_{mode.value}_perf"
        if cache_key in self._performance_cache:
            cache_time, cached_value = self._performance_cache[cache_key]
            if datetime.now() - cache_time < self._cache_expiry:
                return cached_value

        # Filter recent feedback for this domain and mode
        relevant_feedback = [
            f for f in self.feedback_history
            if f.domain == domain and f.optimization_mode == mode
        ]

        if not relevant_feedback:
            # No feedback yet, use neutral factor
            factor = 1.0
        else:
            # Take recent feedback (last 10 samples)
            recent_feedback = relevant_feedback[-10:]

            # Calculate success rate
            success_rate = sum(1 for f in recent_feedback if f.success) / len(recent_feedback)

            # Calculate average scores
            avg_token_reduction = np.mean([f.token_reduction for f in recent_feedback])
            avg_accuracy = np.mean([f.accuracy_score for f in recent_feedback])
            avg_satisfaction = np.mean([f.user_satisfaction for f in recent_feedback])

            # Combine metrics into performance score
            performance_score = (
                success_rate * 0.4 +
                min(avg_token_reduction / 0.5, 1.0) * 0.3 +  # Normalize to 50% target
                avg_accuracy * 0.2 +
                avg_satisfaction * 0.1
            )

            # Convert performance score to factor
            # High performance (>0.8) → factor = 1.0
            # Medium performance (0.5-0.8) → factor = 0.8-1.0
            # Low performance (<0.5) → factor = 1.2+ (more exploration)
            if performance_score > 0.8:
                factor = 1.0
            elif performance_score > 0.5:
                factor = 0.8 + (performance_score - 0.5) * 0.67
            else:
                factor = 1.2 - performance_score * 0.4

        # Cache the result
        self._performance_cache[cache_key] = (datetime.now(), factor)

        return np.clip(factor, 0.5, 2.0)

    def _calculate_exploration_decay(self, domain: str, mode: OptimizationMode) -> float:
        """
        Calculate exploration decay factor.

        exploration_decay = 1 / (1 + 0.1 × iteration_count)
        Starts at 1.0, decays as iterations increase for convergence.
        """
        iteration_count = self.iteration_counts[f"{domain}_{mode.value}"]
        decay = 1.0 / (1.0 + 0.1 * iteration_count)
        return max(0.1, decay)  # Minimum decay to maintain some exploration

    def _get_context_adjustment(self, context_size: int) -> float:
        """
        Adjust exploration based on context size.

        Larger contexts may need more exploration.
        """
        if context_size < 10:
            return 1.2  # More exploration for small contexts
        elif context_size > 100:
            return 0.9  # Less exploration for large contexts
        else:
            return 1.0  # Normal exploration for medium contexts

    def _get_complexity_adjustment(self, query_complexity: float) -> float:
        """
        Adjust base lambda based on query complexity.

        More complex queries may need different lambda values.
        """
        # Complexity 0.0-1.0 maps to lambda multiplier 0.8-1.3
        return 0.8 + query_complexity * 0.5

    def _calculate_confidence(self, domain: str, mode: OptimizationMode) -> float:
        """
        Calculate confidence in lambda calculation.

        Higher confidence with more domain experience and consistent feedback.
        """
        domain_conf = self.domain_confidence[domain]

        # Count relevant feedback
        relevant_feedback = [
            f for f in self.feedback_history
            if f.domain == domain and f.optimization_mode == mode
        ]

        feedback_confidence = min(len(relevant_feedback) / 20.0, 1.0)  # Max at 20 samples

        # Combine domain and feedback confidence
        overall_confidence = (domain_conf + feedback_confidence) / 2.0

        return np.clip(overall_confidence, 0.1, 0.95)

    def _update_domain_factor(self, feedback: PerformanceFeedback) -> None:
        """
        Update domain factor based on performance feedback.

        Uses exponential moving average for gradual learning.
        """
        domain = feedback.domain

        # Calculate feedback score
        feedback_score = (
            (1.0 if feedback.success else 0.0) * 0.4 +
            min(feedback.token_reduction / 0.5, 1.0) * 0.3 +
            feedback.accuracy_score * 0.2 +
            feedback.user_satisfaction * 0.1
        )

        # If performance is poor, we may need to adjust lambda
        if feedback_score < 0.6:
            # Poor performance - adjust domain factor
            if feedback.token_reduction < 0.2:
                # Low token reduction - decrease domain factor (increase lambda)
                adjustment = 1.0 - (0.6 - feedback_score)
            else:
                # Good token reduction but other issues - increase domain factor
                adjustment = 1.0 + (0.6 - feedback_score)
        else:
            # Good performance - small adjustment toward 1.0
            current_factor = self.domain_factors[domain]
            adjustment = 1.0 + (1.0 - current_factor) * 0.1

        # Apply exponential moving average
        current_factor = self.domain_factors[domain]
        new_factor = current_factor * (1 - self.domain_learning_rate) + adjustment * self.domain_learning_rate

        # Ensure bounds
        self.domain_factors[domain] = np.clip(new_factor, 0.5, 2.0)

        # Update domain confidence
        self.domain_confidence[domain] = min(self.domain_confidence[domain] + 0.02, 0.9)

    def export_state(self) -> Dict[str, any]:
        """
        Export tuner state for persistence.

        Returns:
            Dictionary with tuner state
        """
        with self._lock:
            return {
                'domain_factors': dict(self.domain_factors),
                'domain_confidence': dict(self.domain_confidence),
                'iteration_counts': dict(self.iteration_counts),
                'feedback_history': [
                    {
                        'timestamp': f.timestamp.isoformat(),
                        'lambda_used': f.lambda_used,
                        'mode': f.optimization_mode.value,
                        'domain': f.domain,
                        'token_reduction': f.token_reduction,
                        'accuracy_score': f.accuracy_score,
                        'user_satisfaction': f.user_satisfaction,
                        'success': f.success
                    }
                    for f in list(self.feedback_history)[-20:]  # Export last 20 samples
                ],
                'configuration': {
                    'feedback_window_size': self.feedback_window_size,
                    'domain_learning_rate': self.domain_learning_rate,
                    'performance_weight': self.performance_weight
                }
            }

    def import_state(self, state: Dict[str, any]) -> None:
        """
        Import tuner state from persistence.

        Args:
            state: State dictionary from export_state()
        """
        with self._lock:
            # Import domain factors
            if 'domain_factors' in state:
                self.domain_factors.update(state['domain_factors'])

            if 'domain_confidence' in state:
                self.domain_confidence.update(state['domain_confidence'])

            if 'iteration_counts' in state:
                self.iteration_counts.update(state['iteration_counts'])

            # Import feedback history
            if 'feedback_history' in state:
                self.feedback_history.clear()
                for fb_data in state['feedback_history']:
                    feedback = PerformanceFeedback(
                        timestamp=datetime.fromisoformat(fb_data['timestamp']),
                        lambda_used=fb_data['lambda_used'],
                        optimization_mode=OptimizationMode(fb_data['mode']),
                        domain=fb_data['domain'],
                        token_reduction=fb_data['token_reduction'],
                        accuracy_score=fb_data['accuracy_score'],
                        user_satisfaction=fb_data['user_satisfaction'],
                        processing_time_ms=fb_data.get('processing_time_ms', 0.0),
                        success=fb_data['success']
                    )
                    self.feedback_history.append(feedback)

            logger.info("Imported lambda tuner state successfully")


# Global instance for easy access
_lambda_tuner: Optional[DynamicLambdaTuner] = None


def get_lambda_tuner() -> DynamicLambdaTuner:
    """
    Get global lambda tuner instance.

    Returns:
        Singleton DynamicLambdaTuner instance
    """
    global _lambda_tuner
    if _lambda_tuner is None:
        _lambda_tuner = DynamicLambdaTuner()
    return _lambda_tuner