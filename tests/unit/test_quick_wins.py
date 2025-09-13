"""
Unit tests for Quick Wins Optimizer.

Tests all core functionality including duplicate removal, semantic deduplication,
and chunk overlap removal following CEGO testing requirements.
"""

import pytest
from backend.optimizers.quick_wins import QuickWinsOptimizer
from backend.api.quick_wins_api import QuickWinsAPI


class TestQuickWinsOptimizer:
    """Test suite for QuickWinsOptimizer following CEGO patterns."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = QuickWinsOptimizer()
        
        # Standard test data
        self.test_context = [
            "This is a test document about machine learning.",
            "This is a test document about machine learning.",  # Exact duplicate
            "This is a test document about ML.",  # Near duplicate
            "Machine learning is a subset of artificial intelligence.",
            "ML is a subset of AI.",  # Semantic duplicate
            "Python is a programming language used for data science.",
            "Python is used for data science and machine learning.",  # Overlap
            "Deep learning uses neural networks with multiple layers.",
            "Neural networks with multiple layers are used in deep learning."  # Semantic duplicate
        ]
    
    def test_exact_duplicate_removal(self):
        """Test removal of exact duplicates."""
        test_data = [
            "Document 1",
            "Document 1",  # Exact duplicate
            "Document 2",
            "Document 1"   # Another exact duplicate
        ]
        
        result = self.optimizer._remove_exact_duplicates(test_data)
        
        assert len(result) == 2
        assert "Document 1" in result
        assert "Document 2" in result
        assert result.count("Document 1") == 1
    
    def test_near_duplicate_detection(self):
        """Test detection of near duplicates with minor differences."""
        text1 = "the quick brown fox jumps over the lazy dog"
        text2 = "the quick brown fox jumps over lazy dog"  # Missing "the"
        text3 = "completely different text here"
        
        assert self.optimizer._is_near_duplicate(
            self.optimizer._normalize_text(text1),
            self.optimizer._normalize_text(text2)
        )
        
        assert not self.optimizer._is_near_duplicate(
            self.optimizer._normalize_text(text1),
            self.optimizer._normalize_text(text3)
        )
    
    def test_semantic_duplicate_removal(self):
        """Test semantic duplicate detection using word vectors."""
        test_data = [
            "Machine learning is awesome",
            "ML is really awesome",  # High semantic similarity
            "Python programming language",  # Different topic
            "Artificial intelligence and ML are awesome"  # Medium similarity
        ]
        
        result = self.optimizer._remove_semantic_duplicates(test_data)
        
        # Should remove the highly similar second item
        assert len(result) < len(test_data)
        assert "Machine learning is awesome" in result
        assert "Python programming language" in result
    
    def test_chunk_overlap_removal(self):
        """Test removal of overlapping text chunks."""
        test_data = [
            "The quick brown fox jumps over the lazy dog in the forest",
            "brown fox jumps over the lazy dog in the bright forest",  # High overlap
            "Python is a programming language",  # No overlap
            "the lazy dog in the forest sleeps peacefully"  # Medium overlap with first
        ]
        
        result = self.optimizer._remove_chunk_overlaps(test_data)
        
        # Should remove overlapping chunks
        assert len(result) < len(test_data)
        assert "Python is a programming language" in result
    
    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation between word vectors."""
        vec1 = {"machine": 2, "learning": 1, "is": 1, "awesome": 1}
        vec2 = {"machine": 1, "learning": 2, "awesome": 1}  # Similar
        vec3 = {"python": 1, "programming": 1, "language": 1}  # Different
        
        sim1 = self.optimizer._cosine_similarity_simple(vec1, vec2)
        sim2 = self.optimizer._cosine_similarity_simple(vec1, vec3)
        
        assert sim1 > sim2  # More similar vectors should have higher score
        assert 0 <= sim1 <= 1  # Similarity should be normalized
        assert 0 <= sim2 <= 1
    
    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline with metrics."""
        result = self.optimizer.optimize("machine learning", self.test_context)
        
        # Verify result structure follows CEGO format
        assert 'optimized_context' in result
        assert 'stats' in result
        assert 'metadata' in result
        
        # Verify stats structure
        stats = result['stats']
        assert 'original' in stats
        assert 'final' in stats
        assert 'reduction' in stats
        assert 'processing_time_ms' in stats
        assert 'algorithm_used' in stats
        
        # Verify optimization occurred
        assert stats['final']['pieces'] < stats['original']['pieces']
        assert stats['reduction']['token_reduction_pct'] > 0
        
        # Verify target reduction achieved (20-30% for Quick Wins)
        assert stats['reduction']['token_reduction_pct'] >= 0.15  # At least 15%
    
    def test_empty_context_handling(self):
        """Test handling of empty context pool."""
        result = self.optimizer.optimize("test query", [])
        
        assert result['optimized_context'] == []
        assert 'error' in result['stats']
        assert result['stats']['reduction']['token_reduction_pct'] == 0.0
    
    def test_token_limit_enforcement(self):
        """Test that token limits are properly enforced."""
        max_tokens = 50  # Very small limit
        
        result = self.optimizer.optimize(
            "machine learning", 
            self.test_context, 
            max_tokens=max_tokens
        )
        
        final_tokens = result['stats']['final']['tokens']
        assert final_tokens <= max_tokens
    
    def test_fallback_mechanism(self):
        """Test that fallback mechanism works when optimization fails."""
        # Create an optimizer that will trigger fallback
        class FailingOptimizer(QuickWinsOptimizer):
            def _optimize_internal(self, query, context_pool):
                raise Exception("Simulated failure")
        
        failing_optimizer = FailingOptimizer()
        result = failing_optimizer.optimize("test", self.test_context)
        
        # Should still return valid result via fallback
        assert 'optimized_context' in result
        assert result['stats']['algorithm_used'].endswith('_fallback')
        assert result['stats']['fallback_used'] is True


class TestQuickWinsAPI:
    """Test suite for QuickWinsAPI following CEGO patterns."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api = QuickWinsAPI()
    
    def test_valid_optimization_request(self):
        """Test valid optimization request through API."""
        test_context = [
            "Test document 1",
            "Test document 2",
            "Test document 1"  # Duplicate
        ]
        
        result = self.api.optimize("test query", test_context)
        
        # Verify API response format
        assert 'optimized_context' in result
        assert 'stats' in result
        assert 'metadata' in result
        assert result['metadata']['api_version'] == self.api.version
        assert result['metadata']['endpoint'] == 'quick_wins'
    
    def test_input_validation(self):
        """Test API input validation."""
        # Test invalid context_pool type
        result = self.api.optimize("test", "not a list")
        assert 'error' in result['stats']
        
        # Test invalid context_pool items
        result = self.api.optimize("test", [1, 2, 3])
        assert 'error' in result['stats']
        
        # Test invalid query type
        result = self.api.optimize(123, ["doc1", "doc2"])
        assert 'error' in result['stats']
    
    def test_health_check(self):
        """Test health check endpoint."""
        health = self.api.health_check()
        
        assert health['status'] == 'healthy'
        assert 'version' in health
        assert 'optimizer' in health


if __name__ == "__main__":
    pytest.main([__file__])