"""
Shannon Entropy Calculator for CEGO optimization.

This module implements Shannon entropy calculation with adaptive method selection
for different content distribution characteristics.

Mathematical Foundation:
H(X) = -Σ(p_i * log2(p_i)) where p_i is the probability of element i

Features:
- Pure Shannon entropy for balanced distributions
- Cross-entropy for skewed distributions  
- KL divergence for divergence measurement
- Adaptive method selection based on distribution skewness
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
from scipy.stats import skew
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EntropyResult:
    """Result of entropy calculation with metadata."""
    value: float
    method_used: str
    distribution_stats: Dict[str, float]
    confidence: float
    processing_time_ms: float


class ShannonEntropyCalculator:
    """
    Advanced Shannon entropy calculator with adaptive method selection.
    
    This calculator automatically selects the optimal entropy calculation method
    based on the characteristics of the input distribution, providing more
    accurate entropy measurements for different types of content.
    """
    
    def __init__(self):
        self.method_thresholds = {
            'low_skew': 0.5,      # Use Shannon for balanced distributions
            'medium_skew': 2.0,   # Use cross-entropy for moderately skewed
            'high_skew': float('inf')  # Use KL divergence for highly skewed
        }
        
        # Minimum probability to avoid log(0)
        self.epsilon = 1e-10
        
        # Cache for performance
        self._distribution_cache = {}
        
    def calculate_entropy(self, 
                         embeddings: np.ndarray, 
                         method: str = 'adaptive',
                         reference_distribution: Optional[np.ndarray] = None) -> EntropyResult:
        """
        Calculate entropy using specified or adaptive method.
        
        Args:
            embeddings: Input embeddings (N x D matrix)
            method: 'shannon', 'cross_entropy', 'kl_divergence', or 'adaptive'
            reference_distribution: Optional reference for cross-entropy/KL divergence
            
        Returns:
            EntropyResult with entropy value and metadata
            
        Example:
            >>> calculator = ShannonEntropyCalculator()
            >>> embeddings = np.random.rand(100, 768)
            >>> result = calculator.calculate_entropy(embeddings)
            >>> print(f"Entropy: {result.value:.3f} using {result.method_used}")
        """
        import time
        start_time = time.time()
        
        if embeddings.size == 0:
            return EntropyResult(0.0, 'empty', {}, 0.0, 0.0)
        
        # Convert embeddings to probability distribution
        probabilities = self._embeddings_to_probabilities(embeddings)
        
        # Calculate distribution statistics
        distribution_stats = self._analyze_distribution(probabilities)
        
        # Select method if adaptive
        if method == 'adaptive':
            method = self._select_optimal_method(distribution_stats['skewness'])
        
        # Calculate entropy using selected method
        entropy_value = self._calculate_by_method(
            probabilities, method, reference_distribution
        )
        
        # Calculate confidence based on distribution characteristics
        confidence = self._calculate_confidence(distribution_stats, method)
        
        processing_time = (time.time() - start_time) * 1000
        
        return EntropyResult(
            value=entropy_value,
            method_used=method,
            distribution_stats=distribution_stats,
            confidence=confidence,
            processing_time_ms=processing_time
        )
    
    def calculate_content_entropy(self, 
                                 content_pieces: List[str],
                                 embedding_function=None) -> EntropyResult:
        """
        Calculate entropy directly from content pieces.
        
        Args:
            content_pieces: List of text content
            embedding_function: Optional custom embedding function
            
        Returns:
            EntropyResult for the content
        """
        if not content_pieces:
            return EntropyResult(0.0, 'empty', {}, 0.0, 0.0)
        
        # Use simple bag-of-words if no embedding function provided
        if embedding_function is None:
            embeddings = self._simple_content_to_embeddings(content_pieces)
        else:
            embeddings = np.array([embedding_function(piece) for piece in content_pieces])
        
        return self.calculate_entropy(embeddings)
    
    def _embeddings_to_probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Convert embeddings to probability distribution.
        
        Uses cosine similarity matrix to create probability distribution
        based on similarity relationships between embeddings.
        """
        if len(embeddings) == 1:
            return np.array([1.0])
        
        # Calculate pairwise cosine similarities
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        normalized_embeddings = embeddings / norms
        similarity_matrix = normalized_embeddings @ normalized_embeddings.T
        
        # Convert similarities to probabilities using softmax
        # Use mean similarity as the "score" for each embedding
        scores = np.mean(similarity_matrix, axis=1)
        probabilities = self._softmax(scores)
        
        # Ensure no zero probabilities
        probabilities = np.maximum(probabilities, self.epsilon)
        probabilities = probabilities / np.sum(probabilities)
        
        return probabilities
    
    def _simple_content_to_embeddings(self, content_pieces: List[str]) -> np.ndarray:
        """
        Convert content to simple embeddings using word frequency vectors.
        
        This is a fallback when no embedding function is provided.
        """
        # Create vocabulary
        all_words = set()
        for piece in content_pieces:
            words = piece.lower().split()
            all_words.update(words)
        
        vocab = sorted(list(all_words))
        vocab_size = len(vocab)
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        # Create embeddings
        embeddings = np.zeros((len(content_pieces), vocab_size))
        
        for i, piece in enumerate(content_pieces):
            words = piece.lower().split()
            word_counts = Counter(words)
            
            for word, count in word_counts.items():
                if word in word_to_idx:
                    embeddings[i, word_to_idx[word]] = count
        
        # Normalize to unit vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        return embeddings
    
    def _analyze_distribution(self, probabilities: np.ndarray) -> Dict[str, float]:
        """
        Analyze characteristics of probability distribution.
        
        Returns statistics used for adaptive method selection.
        """
        return {
            'skewness': float(skew(probabilities)),
            'entropy_shannon': float(-np.sum(probabilities * np.log2(probabilities + self.epsilon))),
            'effective_rank': float(np.exp(-np.sum(probabilities * np.log(probabilities + self.epsilon)))),
            'max_probability': float(np.max(probabilities)),
            'min_probability': float(np.min(probabilities)),
            'variance': float(np.var(probabilities))
        }
    
    def _select_optimal_method(self, skewness: float) -> str:
        """
        Select optimal entropy calculation method based on distribution skewness.

        Args:
            skewness: Skewness of the probability distribution

        Returns:
            Method name: 'shannon', 'cross_entropy', or 'kl_divergence'
        """
        # Handle NaN skewness (uniform distributions)
        if np.isnan(skewness) or np.isinf(skewness):
            return 'shannon'

        abs_skewness = abs(skewness)

        if abs_skewness < self.method_thresholds['low_skew']:
            return 'shannon'
        elif abs_skewness < self.method_thresholds['medium_skew']:
            return 'cross_entropy'
        else:
            return 'kl_divergence'
    
    def _calculate_by_method(self, 
                           probabilities: np.ndarray, 
                           method: str,
                           reference_distribution: Optional[np.ndarray] = None) -> float:
        """
        Calculate entropy using specified method.
        """
        if method == 'shannon':
            return self._calculate_shannon_entropy(probabilities)
        elif method == 'cross_entropy':
            return self._calculate_cross_entropy(probabilities, reference_distribution)
        elif method == 'kl_divergence':
            return self._calculate_kl_divergence(probabilities, reference_distribution)
        else:
            raise ValueError(f"Unknown entropy method: {method}")
    
    def _calculate_shannon_entropy(self, probabilities: np.ndarray) -> float:
        """
        Calculate classic Shannon entropy: H(X) = -Σ(p_i * log2(p_i))
        """
        # Add epsilon to avoid log(0)
        safe_probabilities = probabilities + self.epsilon
        entropy = -np.sum(probabilities * np.log2(safe_probabilities))
        return float(entropy)
    
    def _calculate_cross_entropy(self, 
                                probabilities: np.ndarray,
                                reference_distribution: Optional[np.ndarray] = None) -> float:
        """
        Calculate cross-entropy: H(p,q) = -Σ(p_i * log2(q_i))
        """
        if reference_distribution is None:
            # Use uniform distribution as reference
            reference_distribution = np.ones_like(probabilities) / len(probabilities)
        
        # Ensure same length
        if len(reference_distribution) != len(probabilities):
            # Resize reference to match probabilities
            reference_distribution = np.ones_like(probabilities) / len(probabilities)
        
        # Add epsilon and normalize
        reference_distribution = reference_distribution + self.epsilon
        reference_distribution = reference_distribution / np.sum(reference_distribution)
        
        cross_entropy = -np.sum(probabilities * np.log2(reference_distribution))
        return float(cross_entropy)
    
    def _calculate_kl_divergence(self, 
                                probabilities: np.ndarray,
                                reference_distribution: Optional[np.ndarray] = None) -> float:
        """
        Calculate KL divergence: D_KL(P||Q) = Σ(p_i * log2(p_i/q_i))
        """
        if reference_distribution is None:
            # Use uniform distribution as reference
            reference_distribution = np.ones_like(probabilities) / len(probabilities)
        
        # Ensure same length
        if len(reference_distribution) != len(probabilities):
            reference_distribution = np.ones_like(probabilities) / len(probabilities)
        
        # Add epsilon and normalize both distributions
        probabilities = probabilities + self.epsilon
        reference_distribution = reference_distribution + self.epsilon
        
        probabilities = probabilities / np.sum(probabilities)
        reference_distribution = reference_distribution / np.sum(reference_distribution)
        
        kl_divergence = np.sum(probabilities * np.log2(probabilities / reference_distribution))
        return float(kl_divergence)
    
    def _calculate_confidence(self, distribution_stats: Dict[str, float], method: str) -> float:
        """
        Calculate confidence in entropy measurement based on distribution characteristics.
        
        Higher confidence for:
        - Balanced distributions (Shannon entropy)
        - Sufficient data points
        - Stable variance
        """
        base_confidence = 0.8
        
        # Adjust based on effective rank (higher rank = more balanced = higher confidence)
        rank_factor = min(distribution_stats['effective_rank'] / len(distribution_stats), 1.0)
        
        # Adjust based on variance (moderate variance = good discrimination)
        variance = distribution_stats['variance']
        variance_factor = 1.0 - abs(variance - 0.1) / 0.1  # Optimal around 0.1
        variance_factor = max(0.1, variance_factor)
        
        # Method-specific adjustments
        method_factors = {
            'shannon': 1.0,
            'cross_entropy': 0.9,
            'kl_divergence': 0.8
        }
        
        confidence = base_confidence * rank_factor * variance_factor * method_factors.get(method, 0.7)
        return max(0.1, min(1.0, confidence))
    
    def _softmax(self, scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Apply softmax transformation to convert scores to probabilities.
        """
        # Subtract max for numerical stability
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores / temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        return probabilities
    
    def get_method_info(self) -> Dict[str, Dict]:
        """
        Get information about available entropy calculation methods.
        
        Returns:
            Dictionary with method information and use cases
        """
        return {
            'shannon': {
                'description': 'Classic Shannon entropy for balanced distributions',
                'formula': 'H(X) = -Σ(p_i * log2(p_i))',
                'best_for': 'Balanced probability distributions',
                'skewness_range': f'|skewness| < {self.method_thresholds["low_skew"]}'
            },
            'cross_entropy': {
                'description': 'Cross-entropy for comparing against reference distribution',
                'formula': 'H(p,q) = -Σ(p_i * log2(q_i))',
                'best_for': 'Moderately skewed distributions',
                'skewness_range': f'{self.method_thresholds["low_skew"]} ≤ |skewness| < {self.method_thresholds["medium_skew"]}'
            },
            'kl_divergence': {
                'description': 'KL divergence for measuring distribution divergence',
                'formula': 'D_KL(P||Q) = Σ(p_i * log2(p_i/q_i))',
                'best_for': 'Highly skewed distributions',
                'skewness_range': f'|skewness| ≥ {self.method_thresholds["medium_skew"]}'
            },
            'adaptive': {
                'description': 'Automatically selects optimal method based on distribution characteristics',
                'formula': 'Dynamic selection based on skewness analysis',
                'best_for': 'General use - automatically adapts to data',
                'skewness_range': 'All ranges (automatic selection)'
            }
        }