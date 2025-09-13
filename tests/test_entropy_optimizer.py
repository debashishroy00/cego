"""
Tests for entropy-based optimizer.

This module tests the entropy optimizer that integrates Shannon and 
multi-dimensional entropy analysis with gradient descent optimization.
"""

import unittest
from typing import List

from backend.optimizers.entropy_optimizer import EntropyOptimizer, EntropyOptimizationResult


class TestEntropyOptimizer(unittest.TestCase):
    """Test cases for entropy optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = EntropyOptimizer()
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer.quick_wins)
        self.assertIsNotNone(self.optimizer.shannon_calculator)
        self.assertIsNotNone(self.optimizer.multidim_entropy)
        
        # Check default parameters
        self.assertEqual(self.optimizer.relevance_weight, 0.7)
        self.assertEqual(self.optimizer.entropy_threshold, 0.1)
        self.assertEqual(self.optimizer.max_iterations, 100)
    
    def test_empty_context_pool(self):
        """Test optimization with empty context pool."""
        result = self.optimizer.optimize("test query", [])
        
        self.assertEqual(len(result["optimized_context"]), 0)
        self.assertEqual(result["stats"]["original"]["pieces"], 0)
        self.assertEqual(result["stats"]["final"]["pieces"], 0)
        self.assertEqual(result["stats"]["reduction"]["token_reduction_pct"], 0.0)
    
    def test_single_context_item(self):
        """Test optimization with single context item."""
        context_pool = ["Single context item for testing"]
        result = self.optimizer.optimize("test query", context_pool)
        
        self.assertEqual(len(result["optimized_context"]), 1)
        self.assertEqual(result["stats"]["original"]["pieces"], 1)
        self.assertEqual(result["stats"]["final"]["pieces"], 1)
        self.assertEqual(result["stats"]["reduction"]["token_reduction_pct"], 0.0)
    
    def test_duplicate_removal(self):
        """Test that duplicate content is removed."""
        context_pool = [
            "Machine learning is powerful",
            "Machine learning is powerful",  # Exact duplicate
            "AI algorithms process data",
            "Machine learning is powerful"   # Another duplicate
        ]
        
        result = self.optimizer.optimize("machine learning", context_pool)
        
        # Should reduce duplicates
        self.assertLess(result["stats"]["final"]["pieces"], result["stats"]["original"]["pieces"])
        self.assertGreater(result["stats"]["reduction"]["token_reduction_pct"], 0.0)
    
    def test_diverse_content_optimization(self):
        """Test optimization with diverse content."""
        context_pool = [
            "Machine learning algorithms process data efficiently using statistical methods",
            "Python programming language offers extensive libraries for data science",
            "Deep learning neural networks can recognize complex patterns in images",
            "JavaScript enables interactive web applications and user interfaces", 
            "Database optimization improves query performance and system scalability",
            "Cloud computing provides scalable infrastructure for modern applications",
            "Artificial intelligence transforms industries through automation",
            "Software engineering best practices ensure maintainable code quality"
        ]
        
        result = self.optimizer.optimize("machine learning", context_pool)
        
        # Should maintain relevant content while reducing tokens
        self.assertGreater(len(result["optimized_context"]), 0)
        self.assertLessEqual(result["stats"]["final"]["pieces"], result["stats"]["original"]["pieces"])
        
        # Check that ML-related content is prioritized
        optimized_text = " ".join(result["optimized_context"]).lower()
        self.assertIn("machine learning", optimized_text)
    
    def test_relevance_preservation(self):
        """Test that relevant content is preserved."""
        context_pool = [
            "Machine learning models require training data for accurate predictions",
            "The weather today is sunny with mild temperatures",
            "Deep learning uses neural networks with multiple hidden layers", 
            "My favorite pizza topping is pepperoni with extra cheese",
            "Artificial intelligence applications include computer vision and NLP",
            "Basketball is a popular sport played worldwide"
        ]
        
        result = self.optimizer.optimize("machine learning AI", context_pool)
        
        # Relevant content should be preserved
        optimized_text = " ".join(result["optimized_context"]).lower()
        self.assertTrue(
            "machine learning" in optimized_text or 
            "deep learning" in optimized_text or
            "artificial intelligence" in optimized_text
        )
        
        # Irrelevant content should be reduced
        self.assertLess("pizza" in optimized_text and "basketball" in optimized_text, True)
    
    def test_optimization_with_analysis(self):
        """Test optimization with detailed analysis."""
        context_pool = [
            "Machine learning algorithms analyze patterns in large datasets",
            "Python libraries like scikit-learn simplify ML implementation",
            "Deep learning networks require significant computational resources", 
            "Data preprocessing is crucial for model performance",
            "Cross-validation helps prevent overfitting in models"
        ]
        
        result = self.optimizer.optimize_with_analysis("machine learning", context_pool)
        
        # Check result structure
        self.assertIsInstance(result, EntropyOptimizationResult)
        self.assertGreater(len(result.optimized_context), 0)
        self.assertGreaterEqual(result.token_reduction_percentage, 0.0)
        self.assertGreater(result.processing_time_ms, 0.0)
        
        # Check entropy analysis
        self.assertIn("total_entropy", result.entropy_analysis)
        self.assertIn("dimension_entropies", result.entropy_analysis)
        self.assertIn("confidence", result.entropy_analysis)
        
        # Check confidence score
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
        
        # Check method used
        self.assertEqual(result.method_used, "entropy_gradient_descent")
    
    def test_fallback_to_quick_wins(self):
        """Test fallback to Quick Wins on entropy calculation failure."""
        # Create scenario that might cause entropy calculation issues
        context_pool = ["", "   ", "\n\n"]  # Empty/whitespace content
        
        result = self.optimizer.optimize("test query", context_pool)
        
        # Should still return a result (via fallback)
        self.assertIsInstance(result, dict)
        self.assertIn("optimized_context", result)
        self.assertIn("stats", result)
    
    def test_content_piece_creation(self):
        """Test creation of ContentPiece objects with metadata."""
        context_pool = [
            "def process_data(data): return clean_data",
            "How do I optimize database queries?",
            "Machine learning models require extensive training data"
        ]
        
        content_pieces = self.optimizer._create_content_pieces(context_pool)
        
        self.assertEqual(len(content_pieces), 3)
        
        # Check that content types are inferred
        self.assertEqual(content_pieces[0].content_type, "code")
        self.assertEqual(content_pieces[1].content_type, "question")
        
        # Check that relationships are extracted
        for piece in content_pieces:
            self.assertIsNotNone(piece.relationships)
            self.assertGreater(len(piece.relationships), 0)
    
    def test_relevance_score_calculation(self):
        """Test relevance score calculation."""
        query = "machine learning algorithms"
        context_pool = [
            "Machine learning algorithms are powerful tools",
            "Deep learning uses neural networks",
            "Python is a programming language",
            "The weather is nice today"
        ]
        
        scores = self.optimizer._calculate_relevance_scores(query, context_pool)
        
        self.assertEqual(len(scores), len(context_pool))
        
        # First item should have highest relevance (exact match)
        self.assertGreater(scores[0], scores[2])  # ML > Python
        self.assertGreater(scores[0], scores[3])  # ML > Weather
    
    def test_text_uniqueness_calculation(self):
        """Test text uniqueness calculation."""
        text = "Machine learning is powerful"
        context_pool = [
            "Machine learning is powerful",
            "Deep learning neural networks",
            "Python programming language",
            "Machine learning algorithms"
        ]
        
        uniqueness = self.optimizer._calculate_text_uniqueness(text, context_pool)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(uniqueness, 0.0)
        self.assertLessEqual(uniqueness, 1.0)
    
    def test_gradient_descent_selection(self):
        """Test gradient descent content selection."""
        query = "machine learning"
        context_pool = [
            "Machine learning algorithms process data",
            "Deep learning neural networks",
            "Python programming basics",
            "Weather forecast information",
            "Artificial intelligence applications"
        ]
        
        # Mock entropy result for testing
        from backend.core.entropy.multi_dimensional import MultiDimensionalResult
        entropy_result = MultiDimensionalResult(
            total_entropy=2.5,
            dimension_entropies={'semantic': 2.0, 'temporal': 0.5, 'relational': 0.0, 'structural': 1.0},
            dimension_weights={'semantic': 0.4, 'temporal': 0.2, 'relational': 0.2, 'structural': 0.2},
            confidence=0.8,
            processing_time_ms=10.0,
            metadata={}
        )
        
        selected_indices = self.optimizer._gradient_descent_selection(
            query, context_pool, entropy_result
        )
        
        # Should return valid indices
        self.assertGreater(len(selected_indices), 0)
        for idx in selected_indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(context_pool))
        
        # Should prioritize relevant content
        self.assertIn(0, selected_indices)  # "Machine learning" content
    
    def test_entropy_info_retrieval(self):
        """Test entropy information retrieval."""
        info = self.optimizer.get_entropy_info()
        
        self.assertIn("shannon_methods", info)
        self.assertIn("multidimensional_info", info)
        self.assertIn("optimization_parameters", info)
        self.assertIn("fallback_strategy", info)
        
        # Check optimization parameters
        params = info["optimization_parameters"]
        self.assertIn("relevance_weight", params)
        self.assertIn("entropy_threshold", params)
        self.assertIn("max_iterations", params)
    
    def test_token_limit_compliance(self):
        """Test that optimizer respects token limits."""
        context_pool = [
            "Machine learning algorithms process large datasets efficiently",
            "Deep learning neural networks learn complex patterns automatically", 
            "Python provides extensive libraries for data science applications",
            "Artificial intelligence transforms industries through intelligent automation",
            "Cloud computing enables scalable infrastructure for modern applications"
        ]
        
        # Test with strict token limit
        result = self.optimizer.optimize("machine learning", context_pool, max_tokens=50)
        
        # Should respect the limit (approximate check)
        total_tokens = sum(len(text.split()) for text in result["optimized_context"])
        self.assertLessEqual(total_tokens, 60)  # Allow some approximation
    
    def test_performance_with_large_context(self):
        """Test performance with larger context pools."""
        # Create larger context pool
        base_contexts = [
            "Machine learning algorithms",
            "Deep learning networks", 
            "Python programming",
            "Data science applications",
            "Artificial intelligence systems"
        ]
        
        # Replicate to create larger pool
        large_context_pool = []
        for i in range(50):
            for ctx in base_contexts:
                large_context_pool.append(f"{ctx} - variation {i}")
        
        import time
        start_time = time.time()
        
        result = self.optimizer.optimize("machine learning", large_context_pool)
        
        processing_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 5 seconds)
        self.assertLess(processing_time, 5.0)
        
        # Should still produce meaningful optimization
        self.assertGreater(result["stats"]["reduction"]["token_reduction_pct"], 0.0)
        self.assertGreater(len(result["optimized_context"]), 0)


if __name__ == '__main__':
    unittest.main()