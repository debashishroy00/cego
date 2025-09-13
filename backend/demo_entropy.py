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

from backend.optimizers.entropy_optimizer import EntropyOptimizer
from backend.optimizers.quick_wins import QuickWinsOptimizer
from datetime import datetime


def demo_entropy_optimization():
    """Demo entropy optimization capabilities."""
    print("=== CEGO Entropy Optimization Demo ===")
    print(f"Timestamp: {datetime.now()}")

    # Sample contexts for demonstration
    test_contexts = [
        {
            "content": "This is a comprehensive documentation explaining the advanced features of our system including detailed API references, extensive examples, and troubleshooting guides.",
            "type": "documentation",
            "priority": "medium"
        },
        {
            "content": "Error: Failed to connect to database",
            "type": "error_log",
            "priority": "high"
        },
        {
            "content": "User clicked button, navigated to page, filled form",
            "type": "user_action",
            "priority": "low"
        }
    ]

    print("\n--- Testing Quick Wins Optimizer ---")
    quick_optimizer = QuickWinsOptimizer()

    for i, context in enumerate(test_contexts):
        print(f"\nTest {i+1}: {context['type']}")
        print(f"Original length: {len(context['content'])} chars")

        result = quick_optimizer.optimize(context['content'])
        print(f"Optimized length: {len(result['optimized_content'])} chars")
        print(f"Compression ratio: {result['metrics']['compression_ratio']:.2f}")
        print(f"Techniques used: {result['techniques_applied']}")

    print("\n--- Testing Entropy Optimizer ---")
    entropy_optimizer = EntropyOptimizer()

    for i, context in enumerate(test_contexts):
        print(f"\nTest {i+1}: {context['type']}")
        print(f"Original length: {len(context['content'])} chars")

        result = entropy_optimizer.optimize(context['content'])
        print(f"Optimized length: {len(result['optimized_content'])} chars")
        print(f"Entropy score: {result['metrics']['entropy_score']:.3f}")
        print(f"Relevance score: {result['metrics']['relevance_score']:.3f}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_entropy_optimization()