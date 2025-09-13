"""
Tests for Shannon entropy calculator.

This module tests the Shannon entropy calculation with adaptive method selection
and multi-dimensional entropy analysis components.
"""

import unittest
import numpy as np
from datetime import datetime, timezone

from backend.core.entropy.shannon_calculator import ShannonEntropyCalculator, EntropyResult
from backend.core.entropy.multi_dimensional import (
    MultiDimensionalEntropy, 
    ContentPiece, 
    content_pieces_from_strings
)


class TestShannonEntropyCalculator(unittest.TestCase):
    """Test cases for Shannon entropy calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = ShannonEntropyCalculator()
    
    def test_empty_input(self):
        """Test entropy calculation with empty input."""
        empty_embeddings = np.array([]).reshape(0, 0)
        result = self.calculator.calculate_entropy(empty_embeddings)
        
        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.method_used, 'empty')
        self.assertEqual(result.confidence, 0.0)
    
    def test_single_embedding(self):
        """Test entropy calculation with single embedding."""
        single_embedding = np.array([[1.0, 0.5, 0.3]])
        result = self.calculator.calculate_entropy(single_embedding)
        
        # Single item should have zero entropy
        self.assertEqual(result.value, 0.0)
        self.assertGreater(result.confidence, 0.0)
    
    def test_identical_embeddings(self):
        """Test entropy calculation with identical embeddings."""
        identical_embeddings = np.array([
            [1.0, 0.5, 0.3],
            [1.0, 0.5, 0.3],
            [1.0, 0.5, 0.3]
        ])
        result = self.calculator.calculate_entropy(identical_embeddings)
        
        # Identical embeddings should have low entropy
        self.assertLess(result.value, 0.1)
        self.assertGreater(result.confidence, 0.0)
    
    def test_diverse_embeddings(self):
        """Test entropy calculation with diverse embeddings."""
        diverse_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0]
        ])
        result = self.calculator.calculate_entropy(diverse_embeddings)
        
        # Diverse embeddings should have higher entropy
        self.assertGreater(result.value, 0.5)
        self.assertIn(result.method_used, ['shannon', 'cross_entropy', 'kl_divergence'])
    
    def test_content_entropy_calculation(self):
        """Test direct content entropy calculation."""
        content_pieces = [
            "Machine learning is transforming technology",
            "AI and ML are revolutionizing software development", 
            "Deep learning networks process complex data patterns",
            "Python is a popular programming language",
            "JavaScript enables interactive web applications"
        ]
        
        result = self.calculator.calculate_content_entropy(content_pieces)
        
        self.assertGreater(result.value, 0.0)
        self.assertIn(result.method_used, ['shannon', 'cross_entropy', 'kl_divergence', 'adaptive'])
        self.assertGreater(result.confidence, 0.0)
        self.assertGreaterEqual(result.processing_time_ms, 0.0)
    
    def test_adaptive_method_selection(self):
        """Test adaptive method selection based on distribution characteristics."""
        # Test with balanced distribution (should use Shannon)
        balanced_embeddings = np.random.rand(50, 10)
        result_balanced = self.calculator.calculate_entropy(balanced_embeddings, method='adaptive')
        
        # Test with skewed distribution
        skewed_embeddings = np.zeros((50, 10))
        skewed_embeddings[:5] = np.random.rand(5, 10)  # Only first few have values
        result_skewed = self.calculator.calculate_entropy(skewed_embeddings, method='adaptive')
        
        # Both should complete successfully
        self.assertGreater(result_balanced.value, 0.0)
        self.assertGreater(result_skewed.value, 0.0)
    
    def test_method_info(self):
        """Test method information retrieval."""
        info = self.calculator.get_method_info()
        
        self.assertIn('shannon', info)
        self.assertIn('cross_entropy', info)
        self.assertIn('kl_divergence', info)
        self.assertIn('adaptive', info)
        
        # Check structure of method info
        shannon_info = info['shannon']
        self.assertIn('description', shannon_info)
        self.assertIn('formula', shannon_info)
        self.assertIn('best_for', shannon_info)


class TestMultiDimensionalEntropy(unittest.TestCase):
    """Test cases for multi-dimensional entropy calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = MultiDimensionalEntropy()
    
    def test_empty_content(self):
        """Test multi-dimensional entropy with empty content."""
        result = self.calculator.calculate_multidimensional_entropy([])
        
        self.assertEqual(result.total_entropy, 0.0)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(len(result.dimension_entropies), 0)
    
    def test_single_content_piece(self):
        """Test with single content piece."""
        content_pieces = [ContentPiece(
            text="Single piece of content",
            timestamp=datetime.now(timezone.utc),
            source="test",
            content_type="text"
        )]
        
        result = self.calculator.calculate_multidimensional_entropy(content_pieces)
        
        # Single piece should have zero or very low entropy
        self.assertGreaterEqual(result.total_entropy, 0.0)
        self.assertLessEqual(result.total_entropy, 0.1)
    
    def test_diverse_content_analysis(self):
        """Test multi-dimensional entropy with diverse content."""
        now = datetime.now(timezone.utc)
        content_pieces = [
            ContentPiece(
                text="Machine learning algorithms process data efficiently",
                timestamp=now,
                source="ml_docs", 
                content_type="documentation",
                relationships=["technical", "algorithmic"]
            ),
            ContentPiece(
                text="def process_data(data): return cleaned_data",
                timestamp=now,
                source="code_repo",
                content_type="code",
                relationships=["function_definition", "data_processing"]
            ),
            ContentPiece(
                text="What are the best practices for data preprocessing?",
                timestamp=now,
                source="forum",
                content_type="question", 
                relationships=["question", "best_practices"]
            ),
            ContentPiece(
                text="Data scientists use Python for analytics and visualization",
                timestamp=now,
                source="blog",
                content_type="article",
                relationships=["professional", "tools"]
            )
        ]
        
        result = self.calculator.calculate_multidimensional_entropy(content_pieces)
        
        # Should have meaningful entropy across dimensions
        self.assertGreater(result.total_entropy, 0.0)
        self.assertGreater(result.confidence, 0.0)
        
        # Check that all dimensions are calculated
        expected_dimensions = ['semantic', 'temporal', 'relational', 'structural']
        for dim in expected_dimensions:
            self.assertIn(dim, result.dimension_entropies)
            self.assertGreaterEqual(result.dimension_entropies[dim], 0.0)
        
        # Check metadata
        self.assertEqual(result.metadata['pieces_analyzed'], 4)
        self.assertIn('dominant_dimension', result.metadata)
    
    def test_semantic_entropy_calculation(self):
        """Test semantic entropy calculation specifically."""
        content_pieces = [
            ContentPiece(text="Python programming language"),
            ContentPiece(text="Java programming language"),
            ContentPiece(text="Machine learning algorithms"),
            ContentPiece(text="Deep learning networks")
        ]
        
        semantic_entropy = self.calculator._calculate_semantic_entropy(content_pieces)
        
        # Should have meaningful semantic entropy
        self.assertGreater(semantic_entropy, 0.0)
    
    def test_temporal_entropy_calculation(self):
        """Test temporal entropy calculation."""
        base_time = datetime.now(timezone.utc)
        content_pieces = [
            ContentPiece(text="First content", timestamp=base_time),
            ContentPiece(text="Second content", timestamp=base_time.replace(hour=base_time.hour + 1)),
            ContentPiece(text="Third content", timestamp=base_time.replace(hour=base_time.hour + 2)),
            ContentPiece(text="Fourth content", timestamp=base_time.replace(hour=base_time.hour + 3))
        ]
        
        temporal_entropy = self.calculator._calculate_temporal_entropy(content_pieces)
        
        # Should have temporal entropy due to time distribution
        self.assertGreater(temporal_entropy, 0.0)
    
    def test_relational_entropy_calculation(self):
        """Test relational entropy calculation."""
        content_pieces = [
            ContentPiece(text="Content 1", relationships=["type_a", "category_1"]),
            ContentPiece(text="Content 2", relationships=["type_b", "category_1"]),
            ContentPiece(text="Content 3", relationships=["type_a", "category_2"]),
            ContentPiece(text="Content 4", relationships=["type_c", "category_2"])
        ]
        
        relational_entropy = self.calculator._calculate_relational_entropy(content_pieces)
        
        # Should have relational entropy due to diverse relationships
        self.assertGreater(relational_entropy, 0.0)
    
    def test_structural_entropy_calculation(self):
        """Test structural entropy calculation."""
        content_pieces = [
            ContentPiece(text="Short", content_type="snippet", source="src1"),
            ContentPiece(text="Medium length content piece", content_type="text", source="src2"),
            ContentPiece(text="Very long content piece with multiple sentences and detailed information.", 
                        content_type="document", source="src1"),
            ContentPiece(text="def function(): pass", content_type="code", source="src3")
        ]
        
        structural_entropy = self.calculator._calculate_structural_entropy(content_pieces)
        
        # Should have structural entropy due to diverse types and sources
        self.assertGreater(structural_entropy, 0.0)
    
    def test_content_pieces_from_strings_utility(self):
        """Test utility function for creating ContentPiece objects."""
        texts = ["Text 1", "Text 2", "Text 3"]
        sources = ["source1", "source2", "source3"]
        
        pieces = content_pieces_from_strings(texts, sources=sources)
        
        self.assertEqual(len(pieces), 3)
        for i, piece in enumerate(pieces):
            self.assertEqual(piece.text, texts[i])
            self.assertEqual(piece.source, sources[i])
            self.assertIsInstance(piece, ContentPiece)
    
    def test_dimension_info(self):
        """Test dimension information retrieval."""
        info = self.calculator.get_dimension_info()
        
        expected_dimensions = ['semantic', 'temporal', 'relational', 'structural']
        for dim in expected_dimensions:
            self.assertIn(dim, info)
            dim_info = info[dim]
            self.assertIn('description', dim_info)
            self.assertIn('measures', dim_info)
            self.assertIn('best_for', dim_info)
            self.assertIn('weight', dim_info)
    
    def test_custom_weights(self):
        """Test multi-dimensional entropy with custom weights."""
        content_pieces = [
            ContentPiece(text="Content 1", content_type="text"),
            ContentPiece(text="Content 2", content_type="code"),
            ContentPiece(text="Content 3", content_type="text")
        ]
        
        custom_weights = {
            'semantic': 0.5,
            'temporal': 0.1,
            'relational': 0.1,
            'structural': 0.3
        }
        
        result = self.calculator.calculate_multidimensional_entropy(
            content_pieces, custom_weights
        )
        
        self.assertEqual(result.dimension_weights, custom_weights)
        self.assertEqual(result.metadata['weight_scheme'], 'custom')


if __name__ == '__main__':
    unittest.main()