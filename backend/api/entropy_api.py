"""
Entropy API handler for advanced CEGO optimization.

This module provides the API interface for entropy-based optimization
using Shannon entropy and multi-dimensional analysis.
"""

import logging
from typing import List, Optional, Dict, Any
import time

from ..optimizers.entropy_optimizer import EntropyOptimizer, EntropyOptimizationResult
from ..core.entropy.shannon_calculator import ShannonEntropyCalculator
from ..core.entropy.multi_dimensional import (
    MultiDimensionalEntropy, 
    content_pieces_from_strings
)

logger = logging.getLogger(__name__)


class EntropyAPI:
    """
    API handler for entropy-based optimization services.
    
    Provides high-level interfaces for entropy calculation and optimization
    with comprehensive error handling and fallback mechanisms.
    """
    
    def __init__(self):
        """Initialize entropy API with optimizers."""
        self.optimizer = EntropyOptimizer()
        self.shannon_calculator = ShannonEntropyCalculator()
        self.multidim_entropy = MultiDimensionalEntropy()
        self.version = "2.0.0-mvp"
        
    def optimize_with_analysis(self, 
                             query: str, 
                             context_pool: List[str],
                             max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize context using entropy analysis with detailed results.
        
        Args:
            query: The query to optimize context for
            context_pool: List of context strings to optimize
            max_tokens: Optional token limit
            
        Returns:
            Comprehensive optimization results with entropy analysis
        """
        try:
            start_time = time.time()
            
            # Validate inputs
            if not context_pool:
                return {
                    "optimized_context": [],
                    "original_count": 0,
                    "optimized_count": 0,
                    "token_reduction_percentage": 0.0,
                    "processing_time_ms": 0.0,
                    "method_used": "entropy_optimization",
                    "entropy_analysis": {},
                    "confidence_score": 0.0,
                    "phase_transitions": [],
                    "error": "Empty context pool provided"
                }
            
            # Perform optimization with analysis
            result = self.optimizer.optimize_with_analysis(
                query=query,
                context_pool=context_pool,
                max_tokens=max_tokens
            )
            
            # Return the result directly - entropy optimizer already returns the correct format
            result["original_count"] = len(context_pool)
            result["optimized_count"] = len(result["optimized_context"])
            return result
            
        except Exception as e:
            logger.error(f"Entropy optimization failed: {e}")
            return {
                "optimized_context": context_pool,  # Return original on error
                "original_count": len(context_pool),
                "optimized_count": len(context_pool),
                "token_reduction_percentage": 0.0,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "method_used": "error_fallback",
                "entropy_analysis": {},
                "confidence_score": 0.0,
                "phase_transitions": [],
                "error": str(e)
            }
    
    def analyze_content_entropy(self, 
                               query: str, 
                               context_pool: List[str]) -> Dict[str, Any]:
        """
        Analyze content entropy without performing optimization.
        
        Args:
            query: The query for relevance analysis
            context_pool: List of context strings to analyze
            
        Returns:
            Detailed entropy analysis results
        """
        try:
            start_time = time.time()
            
            if not context_pool:
                return {
                    "total_content_pieces": 0,
                    "shannon_entropy": 0.0,
                    "multidimensional_entropy": {},
                    "content_distribution": {},
                    "recommendations": ["No content provided for analysis"],
                    "processing_time_ms": 0.0,
                    "error": "Empty context pool provided"
                }
            
            # Create content pieces for analysis
            content_pieces = self.optimizer._create_content_pieces(context_pool)
            
            # Calculate Shannon entropy
            shannon_result = self.shannon_calculator.calculate_content_entropy(context_pool)
            
            # Calculate multi-dimensional entropy
            multidim_result = self.multidim_entropy.calculate_multidimensional_entropy(content_pieces)
            
            # Analyze content distribution
            distribution_analysis = self._analyze_content_distribution(context_pool)
            
            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(
                shannon_result, multidim_result, distribution_analysis
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "total_content_pieces": len(context_pool),
                "shannon_entropy": {
                    "value": shannon_result.value,
                    "method_used": shannon_result.method_used,
                    "confidence": shannon_result.confidence,
                    "distribution_stats": shannon_result.distribution_stats
                },
                "multidimensional_entropy": {
                    "total_entropy": multidim_result.total_entropy,
                    "dimension_entropies": multidim_result.dimension_entropies,
                    "dimension_weights": multidim_result.dimension_weights,
                    "confidence": multidim_result.confidence,
                    "metadata": multidim_result.metadata
                },
                "content_distribution": distribution_analysis,
                "recommendations": recommendations,
                "processing_time_ms": processing_time,
                "analysis_version": self.version
            }
            
        except Exception as e:
            logger.error(f"Entropy analysis failed: {e}")
            return {
                "total_content_pieces": len(context_pool) if context_pool else 0,
                "shannon_entropy": {},
                "multidimensional_entropy": {},
                "content_distribution": {},
                "recommendations": ["Analysis failed - see error details"],
                "processing_time_ms": (time.time() - start_time) * 1000,
                "error": str(e)
            }
    
    def _analyze_content_distribution(self, context_pool: List[str]) -> Dict[str, Any]:
        """Analyze the distribution characteristics of content."""
        if not context_pool:
            return {}
        
        # Length distribution
        lengths = [len(text) for text in context_pool]
        
        # Content type analysis
        content_types = {}
        for text in context_pool:
            content_type = self.optimizer._infer_content_type(text)
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        # Uniqueness analysis
        total_words = set()
        for text in context_pool:
            words = set(text.lower().split())
            total_words.update(words)
        
        return {
            "length_stats": {
                "min_length": min(lengths),
                "max_length": max(lengths),
                "avg_length": sum(lengths) / len(lengths),
                "total_characters": sum(lengths)
            },
            "content_types": content_types,
            "vocabulary_stats": {
                "unique_words": len(total_words),
                "avg_words_per_piece": len(total_words) / len(context_pool)
            },
            "diversity_score": len(set(context_pool)) / len(context_pool)  # Uniqueness ratio
        }
    
    def _generate_optimization_recommendations(self, 
                                             shannon_result, 
                                             multidim_result, 
                                             distribution_analysis) -> List[str]:
        """Generate optimization recommendations based on entropy analysis."""
        recommendations = []
        
        # Shannon entropy recommendations
        if shannon_result.value < 1.0:
            recommendations.append("Low Shannon entropy detected - content appears highly similar")
            recommendations.append("Consider aggressive deduplication for significant token reduction")
        elif shannon_result.value > 4.0:
            recommendations.append("High Shannon entropy detected - content is very diverse")
            recommendations.append("Use selective optimization to preserve important diversity")
        
        # Multi-dimensional entropy recommendations
        dominant_dimension = multidim_result.metadata.get('dominant_dimension')
        if dominant_dimension:
            recommendations.append(f"Dominant entropy dimension: {dominant_dimension}")
            
            if dominant_dimension == 'semantic':
                recommendations.append("Focus on semantic deduplication and topic clustering")
            elif dominant_dimension == 'temporal':
                recommendations.append("Consider time-based optimization and recency weighting")
            elif dominant_dimension == 'structural':
                recommendations.append("Optimize based on content type and format patterns")
            elif dominant_dimension == 'relational':
                recommendations.append("Preserve important document relationships during optimization")
        
        # Distribution-based recommendations
        diversity_score = distribution_analysis.get('diversity_score', 1.0)
        if diversity_score < 0.3:
            recommendations.append("High content duplication detected - excellent optimization potential")
        elif diversity_score > 0.9:
            recommendations.append("Highly unique content - optimization may have limited impact")
        
        # Confidence-based recommendations
        if multidim_result.confidence < 0.5:
            recommendations.append("Low analysis confidence - consider providing more metadata")
            recommendations.append("Results may improve with timestamps, source info, or relationships")
        
        return recommendations
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about available algorithms and methods."""
        return {
            "version": self.version,
            "algorithms": {
                "entropy_optimization": {
                    "description": "Advanced optimization using entropy analysis",
                    "features": [
                        "Shannon entropy calculation",
                        "Multi-dimensional entropy analysis",
                        "Gradient descent optimization",
                        "Relevance preservation"
                    ],
                    "best_for": "Complex content with diverse characteristics"
                }
            },
            "entropy_methods": self.optimizer.get_entropy_info(),
            "fallback_strategy": "Quick Wins optimizer for error cases"
        }