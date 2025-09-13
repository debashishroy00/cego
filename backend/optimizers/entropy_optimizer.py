"""
Entropy-based optimizer integrating Shannon and multi-dimensional entropy analysis.

This optimizer extends the Quick Wins foundation with scientific entropy calculations
for more sophisticated content selection and optimization.

Week 2 MVP Features:
- Shannon entropy-guided content selection  
- Multi-dimensional entropy analysis (semantic, temporal, relational, structural)
- Gradient-based optimization with relevance scoring
- Phase transition detection for optimal stopping
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass

from .base_optimizer import BaseOptimizer
from .quick_wins import QuickWinsOptimizer
from ..core.entropy.shannon_calculator import ShannonEntropyCalculator, EntropyResult
from ..core.entropy.multi_dimensional import (
    MultiDimensionalEntropy, 
    ContentPiece, 
    MultiDimensionalResult,
    content_pieces_from_strings
)

logger = logging.getLogger(__name__)


@dataclass
class EntropyOptimizationResult:
    """Enhanced result including entropy analysis."""
    optimized_context: List[str]
    original_count: int
    optimized_count: int
    token_reduction_percentage: float
    processing_time_ms: float
    entropy_analysis: Dict[str, Any]
    confidence_score: float
    method_used: str
    phase_transitions: List[Dict[str, Any]]


class EntropyOptimizer(BaseOptimizer):
    """
    Advanced optimizer using Shannon and multi-dimensional entropy analysis.
    
    Combines Quick Wins efficiency with scientific entropy calculations
    for superior content selection and optimization performance.
    """
    
    def __init__(self):
        super().__init__()
        self.quick_wins = QuickWinsOptimizer()
        self.shannon_calculator = ShannonEntropyCalculator()
        self.multidim_entropy = MultiDimensionalEntropy()
        
        # Optimization parameters
        self.relevance_weight = 0.7  # Lambda parameter
        self.entropy_threshold = 0.1  # Minimum entropy for inclusion
        self.max_iterations = 100
        self.convergence_threshold = 0.01
        
        # Phase transition tracking
        self.entropy_history = []
        self.transition_threshold = 0.05
        
    def _optimize_internal(self, query: str, context_pool: List[str]) -> List[str]:
        """
        Execute entropy-guided optimization with fallback to Quick Wins.
        
        Strategy:
        1. Apply Quick Wins preprocessing for efficiency
        2. Convert to ContentPiece objects for entropy analysis
        3. Calculate multi-dimensional entropy
        4. Use gradient descent for content selection
        5. Apply phase transition detection
        6. Return optimized context maintaining relevance
        """
        if not context_pool:
            return []
        
        try:
            # Phase 1: Quick Wins preprocessing for efficiency
            preprocessed = self.quick_wins._optimize_internal(query, context_pool)
            
            if len(preprocessed) <= 3:
                # Too few items for entropy analysis, return Quick Wins result
                return preprocessed
            
            # Phase 2: Enhanced entropy analysis
            content_pieces = self._create_content_pieces(preprocessed)
            
            # Phase 3: Multi-dimensional entropy calculation
            entropy_result = self.multidim_entropy.calculate_multidimensional_entropy(content_pieces)
            
            # Phase 4: Gradient-based selection
            selected_indices = self._gradient_descent_selection(
                query, preprocessed, entropy_result
            )
            
            # Phase 5: Apply selections
            optimized_context = [preprocessed[i] for i in selected_indices]
            
            # Ensure minimum content preservation
            if len(optimized_context) < max(1, len(context_pool) * 0.1):
                # Fallback to Quick Wins if too aggressive
                return preprocessed[:max(3, len(context_pool) // 4)]
            
            return optimized_context
            
        except Exception as e:
            logger.warning(f"Entropy optimization failed: {e}. Falling back to Quick Wins.")
            return self.quick_wins._optimize_internal(query, context_pool)
    
    def _create_content_pieces(self, context_pool: List[str]) -> List[ContentPiece]:
        """Convert strings to ContentPiece objects with metadata."""
        current_time = datetime.now(timezone.utc)
        pieces = []
        
        for i, text in enumerate(context_pool):
            piece = ContentPiece(
                text=text,
                timestamp=current_time,
                source=f"context_{i}",
                content_type=self._infer_content_type(text),
                relationships=self._extract_relationships(text, context_pool),
                metadata={"index": i, "length": len(text)}
            )
            pieces.append(piece)
        
        return pieces
    
    def _infer_content_type(self, text: str) -> str:
        """Infer content type from text characteristics."""
        # Check for code patterns first, regardless of length
        if "def " in text or "class " in text or "import " in text or "return " in text:
            return "code"
        elif '?' in text:
            return "question"  
        elif text.count('\n') > 5:
            return "document"
        elif len(text) < 50:
            return "short_snippet"
        else:
            return "text"
    
    def _extract_relationships(self, text: str, context_pool: List[str]) -> List[str]:
        """Extract simple relationship patterns."""
        relationships = []
        
        # Check for code relationships
        if "import" in text:
            relationships.append("import_dependency")
        if "def " in text:
            relationships.append("function_definition")
        if "class " in text:
            relationships.append("class_definition")
        
        # Check for reference relationships
        if any(word in text.lower() for word in ["see", "refer", "mentioned", "above", "below"]):
            relationships.append("reference")
        
        # Default relationship
        if not relationships:
            relationships.append("standalone")
        
        return relationships
    
    def _gradient_descent_selection(self, 
                                  query: str, 
                                  context_pool: List[str],
                                  entropy_result: MultiDimensionalResult) -> List[int]:
        """
        Use gradient descent to select optimal content subset.
        
        Objective: Maximize entropy while maintaining relevance to query.
        """
        n_items = len(context_pool)
        
        # Initialize selection probabilities
        selection_probs = np.ones(n_items) * 0.5
        
        # Calculate relevance scores
        relevance_scores = self._calculate_relevance_scores(query, context_pool)
        
        # Gradient descent iterations
        for iteration in range(self.max_iterations):
            # Calculate gradients
            entropy_gradients = self._calculate_entropy_gradients(
                context_pool, selection_probs, entropy_result
            )
            relevance_gradients = relevance_scores
            
            # Combined gradient with lambda weighting
            combined_gradients = (
                (1 - self.relevance_weight) * entropy_gradients + 
                self.relevance_weight * relevance_gradients
            )
            
            # Update probabilities
            learning_rate = 0.1 * (0.9 ** iteration)  # Decay learning rate
            selection_probs += learning_rate * combined_gradients
            selection_probs = np.clip(selection_probs, 0.01, 0.99)
            
            # Check convergence
            if iteration > 0 and np.mean(np.abs(combined_gradients)) < self.convergence_threshold:
                break
        
        # Convert probabilities to binary selection
        # Select top items based on final probabilities
        n_select = max(1, min(n_items, int(n_items * 0.7)))  # Select 70% or minimum 1
        selected_indices = np.argsort(selection_probs)[-n_select:]
        
        return sorted(selected_indices.tolist())
    
    def _calculate_relevance_scores(self, query: str, context_pool: List[str]) -> np.ndarray:
        """Calculate relevance scores using simple similarity."""
        query_words = set(query.lower().split())
        scores = []
        
        for context in context_pool:
            context_words = set(context.lower().split())
            if not context_words:
                scores.append(0.0)
                continue
            
            # Jaccard similarity
            intersection = len(query_words & context_words)
            union = len(query_words | context_words)
            similarity = intersection / union if union > 0 else 0.0
            scores.append(similarity)
        
        return np.array(scores)
    
    def _calculate_entropy_gradients(self, 
                                   context_pool: List[str],
                                   selection_probs: np.ndarray,
                                   entropy_result: MultiDimensionalResult) -> np.ndarray:
        """
        Calculate gradients for entropy optimization.
        
        Simple approximation: items with higher individual entropy contribution
        get higher gradients.
        """
        gradients = np.zeros(len(context_pool))
        
        # Use semantic entropy as primary gradient signal
        semantic_entropy = entropy_result.dimension_entropies.get('semantic', 0.0)
        
        for i, text in enumerate(context_pool):
            # Simple gradient approximation based on text diversity
            text_length_factor = min(len(text) / 1000, 1.0)  # Normalize length
            uniqueness_factor = self._calculate_text_uniqueness(text, context_pool)
            
            gradients[i] = semantic_entropy * text_length_factor * uniqueness_factor
        
        # Normalize gradients
        if np.max(gradients) > 0:
            gradients = gradients / np.max(gradients)
        
        return gradients
    
    def _calculate_text_uniqueness(self, text: str, context_pool: List[str]) -> float:
        """Calculate how unique this text is compared to others."""
        text_words = set(text.lower().split())
        
        if not text_words:
            return 0.0
        
        # Calculate average similarity to other texts
        similarities = []
        for other_text in context_pool:
            if other_text == text:
                continue
            
            other_words = set(other_text.lower().split())
            if not other_words:
                continue
            
            intersection = len(text_words & other_words)
            union = len(text_words | other_words)
            similarity = intersection / union if union > 0 else 0.0
            similarities.append(similarity)
        
        # Uniqueness is inverse of average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.0
        uniqueness = 1.0 - avg_similarity
        
        return uniqueness
    
    def optimize_with_analysis(self, 
                             query: str, 
                             context_pool: List[str],
                             max_tokens: Optional[int] = None) -> EntropyOptimizationResult:
        """
        Optimize with detailed entropy analysis and metadata.
        
        Returns comprehensive results including entropy breakdown,
        confidence scores, and optimization metadata.
        """
        import time
        start_time = time.time()
        
        # Perform optimization
        result = self.optimize(query, context_pool, max_tokens)
        
        # Calculate additional entropy analysis
        if len(context_pool) > 1:
            content_pieces = self._create_content_pieces(context_pool)
            entropy_analysis = self.multidim_entropy.calculate_multidimensional_entropy(content_pieces)
        else:
            entropy_analysis = MultiDimensionalResult(0.0, {}, {}, 0.0, 0.0, {})
        
        processing_time = (time.time() - start_time) * 1000
        
        return EntropyOptimizationResult(
            optimized_context=result["optimized_context"],
            original_count=result["stats"]["original"]["pieces"],
            optimized_count=result["stats"]["final"]["pieces"],
            token_reduction_percentage=result["stats"]["reduction"]["token_reduction_pct"],
            processing_time_ms=processing_time,
            entropy_analysis={
                "total_entropy": entropy_analysis.total_entropy,
                "dimension_entropies": entropy_analysis.dimension_entropies,
                "dimension_weights": entropy_analysis.dimension_weights,
                "confidence": entropy_analysis.confidence,
                "metadata": entropy_analysis.metadata
            },
            confidence_score=entropy_analysis.confidence,
            method_used="entropy_gradient_descent",
            phase_transitions=[]  # TODO: Implement phase transition detection
        )
    
    def get_entropy_info(self) -> Dict[str, Any]:
        """Get information about entropy calculation methods and parameters."""
        return {
            "shannon_methods": self.shannon_calculator.get_method_info(),
            "multidimensional_info": self.multidim_entropy.get_dimension_info(),
            "optimization_parameters": {
                "relevance_weight": self.relevance_weight,
                "entropy_threshold": self.entropy_threshold,
                "max_iterations": self.max_iterations,
                "convergence_threshold": self.convergence_threshold
            },
            "fallback_strategy": "Quick Wins optimizer for error cases and small datasets"
        }