"""
Pattern Recognition API - Simple HTTP interface for Phase 1 optimization.

This module provides a clean API interface for the Pattern Recognition optimizer,
following CEGO standards for input/output format and error handling.
"""

from typing import List, Dict, Optional
import logging

from ..optimizers.pattern_recognition import PatternRecognitionOptimizer

logger = logging.getLogger(__name__)


class PatternRecognitionAPI:
    """
    API wrapper for Pattern Recognition optimization.

    Provides simple HTTP-ready interface with proper error handling,
    validation, and standardized response format.
    """
    
    def __init__(self):
        self.optimizer = PatternRecognitionOptimizer()
        self.version = "1.0.0"
    
    def optimize(self, query: str, context_pool: List[str], 
                 max_tokens: Optional[int] = None) -> Dict:
        """
        Main API endpoint for context optimization.
        
        Args:
            query: The user query
            context_pool: List of context pieces to optimize
            max_tokens: Optional token limit for final result
            
        Returns:
            Standardized CEGO response with optimized context and metrics
            
        Example:
            >>> api = PatternRecognitionAPI()
            >>> result = api.optimize("machine learning", ["doc1", "doc2"])
            >>> print(f"Saved {result['stats']['reduction']['token_reduction_pct']:.1%}")
            Saved 32.5%
        """
        # Input validation
        if not isinstance(context_pool, list):
            return self._error_response("context_pool must be a list")
        
        if not all(isinstance(item, str) for item in context_pool):
            return self._error_response("All context_pool items must be strings")
        
        if not isinstance(query, str):
            return self._error_response("query must be a string")
        
        # Apply optimization
        try:
            result = self.optimizer.optimize(query, context_pool, max_tokens)
            
            # Add API metadata
            result['metadata']['api_version'] = self.version
            result['metadata']['endpoint'] = 'pattern_recognition'
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self._error_response(f"Optimization failed: {str(e)}")
    
    def health_check(self) -> Dict:
        """
        Health check endpoint for monitoring.
        
        Returns:
            System health status
        """
        return {
            'status': 'healthy',
            'version': self.version,
            'optimizer': self.optimizer.algorithm_name,
            'optimizer_version': self.optimizer.version
        }
    
    def _error_response(self, error_msg: str) -> Dict:
        """
        Generate standardized error response.
        
        Args:
            error_msg: Error message
            
        Returns:
            Error response in CEGO format
        """
        return {
            'optimized_context': [],
            'stats': {
                'error': error_msg,
                'original': {'pieces': 0, 'tokens': 0},
                'final': {'pieces': 0, 'tokens': 0},
                'reduction': {'token_reduction_pct': 0.0, 'pieces_saved': 0},
                'processing_time_ms': 0.0,
                'algorithm_used': 'error',
                'phase_transitions': []
            },
            'metadata': {
                'version': self.version,
                'api_version': self.version,
                'endpoint': 'pattern_recognition',
                'rollback_available': False
            }
        }