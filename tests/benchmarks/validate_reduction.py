"""
CEGO Performance Validation Script.

This script validates that optimizers meet performance targets and
can be used for continuous integration checks.
"""

import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.optimizers.quick_wins import QuickWinsOptimizer
from src.validators.golden_queries import GoldenQueryValidator


def main():
    """Run validation benchmark."""
    print("CEGO Validation Benchmark")
    print("=" * 40)
    
    # Initialize optimizer and validator
    optimizer = QuickWinsOptimizer()
    validator = GoldenQueryValidator()
    
    print(f"Testing optimizer: {optimizer.algorithm_name} v{optimizer.version}")
    print(f"Golden queries: {len(validator.golden_queries)} test cases")
    print("")
    
    # Run validation
    start_time = time.time()
    results = validator.run_golden_queries(optimizer)
    validation_time = time.time() - start_time
    
    # Generate and display report
    report = validator.generate_report(results)
    print(report)
    
    print(f"Validation completed in {validation_time:.2f} seconds")
    
    # Exit with appropriate code
    if results['overall_passed']:
        print("\n✅ All validations PASSED")
        sys.exit(0)
    else:
        print("\n❌ Some validations FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()