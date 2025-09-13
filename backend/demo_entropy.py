#!/usr/bin/env python3
"""
Demo script for Week 2 entropy-based optimization.

This script demonstrates the advanced entropy algorithms including:
- Shannon entropy calculation with adaptive method selection
- Multi-dimensional entropy analysis (semantic, temporal, relational, structural)
- Gradient descent optimization with relevance scoring
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.optimizers.entropy_optimizer import EntropyOptimizer
from src.optimizers.quick_wins import QuickWinsOptimizer
from datetime import datetime


def demo_entropy_optimization():
    """Demonstrate entropy-based optimization vs Quick Wins."""
    print("=== CEGO Week 2: Entropy-Based Optimization Demo ===")
    print()
    
    # Create sample context with various characteristics
    context_pool = [
        "Machine learning algorithms analyze patterns in large datasets to make predictions",
        "Python programming language offers extensive libraries for data science and ML",
        "Machine learning algorithms analyze patterns in large datasets to make predictions",  # Duplicate
        "Deep learning neural networks use multiple layers to process complex data",
        "JavaScript enables interactive web applications with real-time user interfaces",
        "Database optimization techniques improve query performance and system scalability", 
        "Artificial intelligence transforms industries through intelligent process automation",
        "Software engineering best practices ensure maintainable and scalable code quality",
        "def train_model(data): return ml_algorithm.fit(data)",  # Code snippet
        "How can I optimize my machine learning model performance?",  # Question
        "Cloud computing provides on-demand scalable infrastructure for applications",
        "Data preprocessing and feature engineering are crucial for model accuracy",
        "Cross-validation helps prevent overfitting in machine learning models"
    ]
    
    query = "machine learning model optimization techniques"
    
    print(f"Query: '{query}'")
    print(f"Original context pool: {len(context_pool)} items")
    print()
    
    # Test Quick Wins (baseline)
    print("--- Quick Wins Optimization (Week 1) ---")
    quick_wins = QuickWinsOptimizer()
    quick_result = quick_wins.optimize(query, context_pool)
    
    print(f"Optimized count: {quick_result['stats']['final']['pieces']}")
    print(f"Token reduction: {quick_result['stats']['reduction']['token_reduction_pct']:.1%}")
    print(f"Processing time: {quick_result['stats']['processing_time_ms']:.1f}ms")
    print()
    
    # Test Entropy Optimization (Week 2)
    print("--- Entropy-Based Optimization (Week 2) ---")
    entropy_optimizer = EntropyOptimizer()
    entropy_result = entropy_optimizer.optimize_with_analysis(query, context_pool)
    
    print(f"Optimized count: {entropy_result.optimized_count}")
    print(f"Token reduction: {entropy_result.token_reduction_percentage:.1%}")
    print(f"Processing time: {entropy_result.processing_time_ms:.1f}ms")
    print(f"Confidence score: {entropy_result.confidence_score:.2f}")
    print(f"Method used: {entropy_result.method_used}")
    print()
    
    # Show entropy analysis breakdown
    print("--- Entropy Analysis Breakdown ---")
    entropy_analysis = entropy_result.entropy_analysis
    print(f"Total entropy: {entropy_analysis['total_entropy']:.3f}")
    print("Dimension entropies:")
    for dim, value in entropy_analysis['dimension_entropies'].items():
        weight = entropy_analysis['dimension_weights'][dim]
        print(f"  {dim}: {value:.3f} (weight: {weight:.1f})")
    print()
    
    # Show optimized context
    print("--- Optimized Context (Entropy-Based) ---")
    for i, context in enumerate(entropy_result.optimized_context, 1):
        content_type = "Code" if "def " in context else "Question" if "?" in context else "Text"
        print(f"{i}. [{content_type}] {context[:80]}{'...' if len(context) > 80 else ''}")
    print()
    
    # Show algorithm information
    print("--- Algorithm Information ---")
    entropy_info = entropy_optimizer.get_entropy_info()
    print(f"Shannon methods available: {list(entropy_info['shannon_methods'].keys())}")
    print(f"Multi-dimensional info: {list(entropy_info['multidimensional_info'].keys())}")
    print(f"Relevance weight: {entropy_info['optimization_parameters']['relevance_weight']}")
    print()
    
    # Performance comparison
    print("--- Performance Comparison ---")
    print(f"Quick Wins:    {quick_result['stats']['reduction']['token_reduction_pct']:.1%} reduction in {quick_result['stats']['processing_time_ms']:.1f}ms")
    print(f"Entropy-based: {entropy_result.token_reduction_percentage:.1%} reduction in {entropy_result.processing_time_ms:.1f}ms")
    
    improvement = entropy_result.token_reduction_percentage - quick_result['stats']['reduction']['token_reduction_pct']
    print(f"Improvement: {improvement:.1%} better token reduction")
    print()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    demo_entropy_optimization()