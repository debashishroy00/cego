"""
Tests for entropy API handler.

This module tests the entropy API that provides high-level interfaces
for entropy calculation and optimization services.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.api.entropy_api import EntropyAPI


class TestEntropyAPI(unittest.TestCase):
    """Test cases for entropy API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api = EntropyAPI()
    
    def test_initialization(self):
        """Test API initialization."""
        self.assertIsNotNone(self.api.optimizer)
        self.assertIsNotNone(self.api.shannon_calculator)
        self.assertIsNotNone(self.api.multidim_entropy)
        self.assertEqual(self.api.version, "2.0.0-mvp")
    
    def test_optimize_with_analysis_empty_context(self):
        """Test optimization with empty context pool."""
        result = self.api.optimize_with_analysis("test query", [])
        
        self.assertEqual(len(result["optimized_context"]), 0)
        self.assertEqual(result["original_count"], 0)
        self.assertEqual(result["optimized_count"], 0)
        self.assertEqual(result["token_reduction_percentage"], 0.0)
        self.assertIn("error", result)
    
    def test_optimize_with_analysis_success(self):
        """Test successful optimization with analysis."""
        context_pool = [
            "Machine learning algorithms process data efficiently",
            "Deep learning networks learn complex patterns",
            "Python provides libraries for data science",
            "Artificial intelligence transforms industries"
        ]
        
        result = self.api.optimize_with_analysis("machine learning", context_pool)
        
        # Check required fields
        self.assertIn("optimized_context", result)
        self.assertIn("original_count", result)
        self.assertIn("optimized_count", result)
        self.assertIn("token_reduction_percentage", result)
        self.assertIn("processing_time_ms", result)
        self.assertIn("method_used", result)
        self.assertIn("entropy_analysis", result)
        self.assertIn("confidence_score", result)
        self.assertIn("stats", result)
        
        # Check values
        self.assertGreater(len(result["optimized_context"]), 0)
        self.assertEqual(result["original_count"], 4)
        self.assertGreaterEqual(result["token_reduction_percentage"], 0.0)
        self.assertGreater(result["processing_time_ms"], 0.0)
        self.assertEqual(result["method_used"], "entropy_gradient_descent")
        
        # Check entropy analysis structure
        entropy_analysis = result["entropy_analysis"]
        self.assertIn("total_entropy", entropy_analysis)
        self.assertIn("dimension_entropies", entropy_analysis)
        self.assertIn("confidence", entropy_analysis)
        
        # Check stats
        stats = result["stats"]
        self.assertEqual(stats["algorithm"], "entropy_optimization")
        self.assertEqual(stats["version"], "2.0.0-mvp")
    
    @patch('src.api.entropy_api.EntropyOptimizer')
    def test_optimize_with_analysis_error_handling(self, mock_optimizer_class):
        """Test error handling in optimization."""
        # Mock optimizer to raise an exception
        mock_optimizer = MagicMock()
        mock_optimizer.optimize_with_analysis.side_effect = Exception("Test error")
        mock_optimizer_class.return_value = mock_optimizer
        
        api = EntropyAPI()
        result = api.optimize_with_analysis("test query", ["test content"])
        
        # Should return error response
        self.assertIn("error", result)
        self.assertEqual(result["method_used"], "error_fallback")
        self.assertEqual(result["optimized_context"], ["test content"])  # Original content
    
    def test_analyze_content_entropy_empty_context(self):
        """Test entropy analysis with empty context."""
        result = self.api.analyze_content_entropy("test query", [])
        
        self.assertEqual(result["total_content_pieces"], 0)
        self.assertEqual(result["shannon_entropy"], 0.0)
        self.assertEqual(result["multidimensional_entropy"], {})
        self.assertIn("error", result)
    
    def test_analyze_content_entropy_success(self):
        """Test successful content entropy analysis."""
        context_pool = [
            "Machine learning algorithms process data efficiently",
            "Deep learning networks learn from examples",
            "Python programming language for data science",
            "Artificial intelligence applications in industry"
        ]
        
        result = self.api.analyze_content_entropy("machine learning", context_pool)
        
        # Check required fields
        self.assertIn("total_content_pieces", result)
        self.assertIn("shannon_entropy", result)
        self.assertIn("multidimensional_entropy", result)
        self.assertIn("content_distribution", result)
        self.assertIn("recommendations", result)
        self.assertIn("processing_time_ms", result)
        
        # Check values
        self.assertEqual(result["total_content_pieces"], 4)
        self.assertGreater(result["processing_time_ms"], 0.0)
        
        # Check Shannon entropy structure
        shannon_entropy = result["shannon_entropy"]
        self.assertIn("value", shannon_entropy)
        self.assertIn("method_used", shannon_entropy)
        self.assertIn("confidence", shannon_entropy)
        self.assertGreaterEqual(shannon_entropy["value"], 0.0)
        
        # Check multi-dimensional entropy structure
        multidim_entropy = result["multidimensional_entropy"]
        self.assertIn("total_entropy", multidim_entropy)
        self.assertIn("dimension_entropies", multidim_entropy)
        self.assertIn("confidence", multidim_entropy)
        
        # Check content distribution
        distribution = result["content_distribution"]
        self.assertIn("length_stats", distribution)
        self.assertIn("content_types", distribution)
        self.assertIn("vocabulary_stats", distribution)
        self.assertIn("diversity_score", distribution)
        
        # Check recommendations
        recommendations = result["recommendations"]
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
    
    def test_content_distribution_analysis(self):
        """Test content distribution analysis."""
        context_pool = [
            "Short text",
            "This is a medium length text with several words",
            "def process_data(data): return cleaned_data",
            "Very long content piece with multiple sentences and detailed information about various topics"
        ]
        
        distribution = self.api._analyze_content_distribution(context_pool)
        
        # Check length statistics
        length_stats = distribution["length_stats"]
        self.assertIn("min_length", length_stats)
        self.assertIn("max_length", length_stats)
        self.assertIn("avg_length", length_stats)
        self.assertIn("total_characters", length_stats)
        
        self.assertGreater(length_stats["max_length"], length_stats["min_length"])
        self.assertGreater(length_stats["avg_length"], 0)
        
        # Check content types
        content_types = distribution["content_types"]
        self.assertIsInstance(content_types, dict)
        self.assertGreater(len(content_types), 0)
        
        # Check vocabulary stats
        vocab_stats = distribution["vocabulary_stats"]
        self.assertIn("unique_words", vocab_stats)
        self.assertIn("avg_words_per_piece", vocab_stats)
        self.assertGreater(vocab_stats["unique_words"], 0)
        
        # Check diversity score
        self.assertIn("diversity_score", distribution)
        self.assertGreaterEqual(distribution["diversity_score"], 0.0)
        self.assertLessEqual(distribution["diversity_score"], 1.0)
    
    def test_optimization_recommendations_generation(self):
        """Test generation of optimization recommendations."""
        # Mock entropy results
        from src.core.entropy.shannon_calculator import EntropyResult
        from src.core.entropy.multi_dimensional import MultiDimensionalResult
        
        shannon_result = EntropyResult(
            value=2.5,
            method_used="shannon",
            distribution_stats={"skewness": 0.3},
            confidence=0.8,
            processing_time_ms=10.0
        )
        
        multidim_result = MultiDimensionalResult(
            total_entropy=3.2,
            dimension_entropies={"semantic": 2.0, "temporal": 0.8, "relational": 0.2, "structural": 0.2},
            dimension_weights={"semantic": 0.4, "temporal": 0.2, "relational": 0.2, "structural": 0.2},
            confidence=0.9,
            processing_time_ms=15.0,
            metadata={"dominant_dimension": "semantic"}
        )
        
        distribution_analysis = {"diversity_score": 0.6}
        
        recommendations = self.api._generate_optimization_recommendations(
            shannon_result, multidim_result, distribution_analysis
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Should mention dominant dimension
        recommendation_text = " ".join(recommendations)
        self.assertIn("semantic", recommendation_text)
    
    def test_algorithm_info_retrieval(self):
        """Test algorithm information retrieval."""
        info = self.api.get_algorithm_info()
        
        self.assertIn("version", info)
        self.assertIn("algorithms", info)
        self.assertIn("entropy_methods", info)
        self.assertIn("fallback_strategy", info)
        
        # Check algorithm information
        algorithms = info["algorithms"]
        self.assertIn("entropy_optimization", algorithms)
        
        entropy_opt = algorithms["entropy_optimization"]
        self.assertIn("description", entropy_opt)
        self.assertIn("features", entropy_opt)
        self.assertIn("best_for", entropy_opt)
        
        # Check that features are listed
        features = entropy_opt["features"]
        self.assertIn("Shannon entropy calculation", features)
        self.assertIn("Multi-dimensional entropy analysis", features)
    
    @patch('src.api.entropy_api.logger')
    def test_error_logging(self, mock_logger):
        """Test that errors are properly logged."""
        # Force an error by mocking optimizer failure
        with patch.object(self.api.optimizer, 'optimize_with_analysis') as mock_optimize:
            mock_optimize.side_effect = Exception("Test error")
            
            result = self.api.optimize_with_analysis("test", ["content"])
            
            # Check that error was logged
            mock_logger.error.assert_called_once()
            self.assertIn("error", result)
    
    def test_recommendations_for_low_entropy(self):
        """Test recommendations for low entropy content."""
        from src.core.entropy.shannon_calculator import EntropyResult
        from src.core.entropy.multi_dimensional import MultiDimensionalResult
        
        # Low entropy scenario
        shannon_result = EntropyResult(
            value=0.5,  # Low entropy
            method_used="shannon",
            distribution_stats={"skewness": 0.1},
            confidence=0.8,
            processing_time_ms=5.0
        )
        
        multidim_result = MultiDimensionalResult(
            total_entropy=0.8,
            dimension_entropies={"semantic": 0.5, "temporal": 0.1, "relational": 0.1, "structural": 0.1},
            dimension_weights={"semantic": 0.4, "temporal": 0.2, "relational": 0.2, "structural": 0.2},
            confidence=0.6,
            processing_time_ms=8.0,
            metadata={"dominant_dimension": "semantic"}
        )
        
        distribution_analysis = {"diversity_score": 0.2}  # Low diversity
        
        recommendations = self.api._generate_optimization_recommendations(
            shannon_result, multidim_result, distribution_analysis
        )
        
        recommendation_text = " ".join(recommendations)
        self.assertIn("low", recommendation_text.lower())
        self.assertIn("deduplication", recommendation_text.lower())
    
    def test_recommendations_for_high_entropy(self):
        """Test recommendations for high entropy content."""
        from src.core.entropy.shannon_calculator import EntropyResult
        from src.core.entropy.multi_dimensional import MultiDimensionalResult
        
        # High entropy scenario
        shannon_result = EntropyResult(
            value=4.5,  # High entropy
            method_used="shannon",
            distribution_stats={"skewness": 0.2},
            confidence=0.9,
            processing_time_ms=12.0
        )
        
        multidim_result = MultiDimensionalResult(
            total_entropy=5.2,
            dimension_entropies={"semantic": 3.0, "temporal": 1.0, "relational": 0.6, "structural": 0.6},
            dimension_weights={"semantic": 0.4, "temporal": 0.2, "relational": 0.2, "structural": 0.2},
            confidence=0.95,
            processing_time_ms=18.0,
            metadata={"dominant_dimension": "semantic"}
        )
        
        distribution_analysis = {"diversity_score": 0.95}  # High diversity
        
        recommendations = self.api._generate_optimization_recommendations(
            shannon_result, multidim_result, distribution_analysis
        )
        
        recommendation_text = " ".join(recommendations)
        self.assertIn("high", recommendation_text.lower())
        self.assertIn("diverse", recommendation_text.lower())


if __name__ == '__main__':
    unittest.main()