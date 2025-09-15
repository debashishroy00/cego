"""
Base optimizer abstract class for all CEGO optimizers.

This module defines the standard interface and patterns that all CEGO optimizers
must follow, ensuring consistency in metrics, error handling, and rollback support.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """
    Abstract base class for all CEGO optimizers.
    
    All optimizers follow this pattern to ensure consistent behavior,
    metrics reporting, and rollback capabilities.
    """
    
    def __init__(self):
        self.version = "1.0.0"
        self.algorithm_name = self.__class__.__name__
        
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.
        
        Args:
            text: Input text to estimate tokens for
            
        Returns:
            Estimated number of tokens (using 4 chars per token ratio)
        """
        return len(text) // 4
    
    @abstractmethod
    def _optimize_internal(self, query: str, context_pool: List[str]) -> List[str]:
        """
        Internal optimization logic - must be implemented by subclasses.
        
        Args:
            query: The user query
            context_pool: List of context pieces to optimize
            
        Returns:
            Optimized list of context pieces
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    def optimize(self, query: str, context_pool: List[str], 
                 max_tokens: Optional[int] = None) -> Dict:
        """
        Main entry point - orchestrates the optimization algorithm.
        
        Follows CEGO standard pattern:
        1. Validate inputs
        2. Apply optimization  
        3. Track metrics
        4. Return results with stats
        
        Args:
            query: The user query
            context_pool: List of context pieces to optimize
            max_tokens: Optional token limit for final result
            
        Returns:
            Dict with optimized_context, stats, and metadata following
            CEGO standard format
            
        Example:
            >>> optimizer = ConcreteOptimizer()
            >>> result = optimizer.optimize("query", ["doc1", "doc2"])
            >>> print(result['stats']['reduction']['token_reduction_pct'])
            0.35
        """
        start_time = time.time()
        
        # 1. Validate inputs
        if not context_pool:
            return self._empty_result("Empty context pool")
            
        if not query.strip():
            return self._empty_result("Empty query")
        
        # 2. Store original metrics
        original_pieces = len(context_pool)
        original_tokens = sum(self.estimate_tokens(text) for text in context_pool)
        
        try:
            # 3. Apply optimization
            optimized_context = self._optimize_internal(query, context_pool)
            
            # 4. Apply token limit if specified
            if max_tokens:
                optimized_context = self._truncate_to_tokens(optimized_context, max_tokens)
            
            # 5. Calculate final metrics
            final_pieces = len(optimized_context)
            final_tokens = sum(self.estimate_tokens(text) for text in optimized_context)
            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            # 6. Calculate semantic retention
            from ..utils.relevance import semantic_retention
            retention = semantic_retention(context_pool, optimized_context)

            # 7. Track metrics and return results
            return {
                'optimized_context': optimized_context,
                'semantic_retention': retention,
                'stats': {
                    'original': {
                        'pieces': original_pieces,
                        'tokens': original_tokens
                    },
                    'final': {
                        'pieces': final_pieces,
                        'tokens': final_tokens
                    },
                    'reduction': {
                        'token_reduction_pct': 1.0 - (final_tokens / original_tokens) if original_tokens > 0 else 0.0,
                        'pieces_saved': original_pieces - final_pieces
                    },
                    'processing_time_ms': processing_time,
                    'algorithm_used': self.algorithm_name,
                    'phase_transitions': []  # Will be populated by advanced optimizers
                },
                'metadata': {
                    'version': self.version,
                    'timestamp': datetime.now().isoformat(),
                    'rollback_available': True
                }
            }
            
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            return self._fallback_optimization(query, context_pool, original_pieces, original_tokens)
    
    def _truncate_to_tokens(self, context_list: List[str], max_tokens: int) -> List[str]:
        """
        Truncate context list to fit within token budget.
        
        Args:
            context_list: List of context pieces
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Truncated list of context pieces
        """
        selected = []
        tokens_used = 0
        
        for ctx in context_list:
            ctx_tokens = self.estimate_tokens(ctx)
            if tokens_used + ctx_tokens <= max_tokens:
                selected.append(ctx)
                tokens_used += ctx_tokens
            else:
                break
        
        return selected
    
    def _empty_result(self, error_msg: str) -> Dict:
        """
        Return empty result with error message.
        
        Args:
            error_msg: Description of the error
            
        Returns:
            Standard CEGO result format with empty context
        """
        return {
            'optimized_context': [],
            'stats': {
                'original': {'pieces': 0, 'tokens': 0},
                'final': {'pieces': 0, 'tokens': 0},
                'reduction': {'token_reduction_pct': 0.0, 'pieces_saved': 0},
                'processing_time_ms': 0.0,
                'algorithm_used': self.algorithm_name,
                'error': error_msg,
                'phase_transitions': []
            },
            'metadata': {
                'version': self.version,
                'timestamp': datetime.now().isoformat(),
                'rollback_available': False
            }
        }
    
    def _fallback_optimization(self, query: str, context_pool: List[str], 
                              original_pieces: int, original_tokens: int) -> Dict:
        """
        Fallback optimization using simple truncation.
        
        This method provides a guaranteed working fallback when the main
        optimization fails, following the "always have a fallback" pattern.
        
        Args:
            query: The user query
            context_pool: Original context pool
            original_pieces: Number of original pieces
            original_tokens: Number of original tokens
            
        Returns:
            Result using simple truncation fallback
        """
        # Simple relevance-based truncation as fallback
        query_terms = set(query.lower().split())
        scored = []
        
        for ctx in context_pool:
            ctx_terms = set(ctx.lower().split())
            overlap = len(query_terms & ctx_terms)
            scored.append((overlap, ctx))
        
        # Sort by relevance and take top pieces
        scored.sort(reverse=True, key=lambda x: x[0])
        fallback_context = [ctx for _, ctx in scored[:min(10, len(scored))]]
        
        final_tokens = sum(self.estimate_tokens(text) for text in fallback_context)
        
        return {
            'optimized_context': fallback_context,
            'stats': {
                'original': {'pieces': original_pieces, 'tokens': original_tokens},
                'final': {'pieces': len(fallback_context), 'tokens': final_tokens},
                'reduction': {
                    'token_reduction_pct': 1.0 - (final_tokens / original_tokens) if original_tokens > 0 else 0.0,
                    'pieces_saved': original_pieces - len(fallback_context)
                },
                'processing_time_ms': 0.0,  # Minimal processing time
                'algorithm_used': f"{self.algorithm_name}_fallback",
                'fallback_used': True,
                'phase_transitions': []
            },
            'metadata': {
                'version': self.version,
                'timestamp': datetime.now().isoformat(),
                'rollback_available': True
            }
        }