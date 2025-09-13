"""
Golden Query Validation Framework.

This module provides validation against a set of golden queries to ensure
optimizations maintain quality while reducing tokens.
"""

from typing import List, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)


class GoldenQueryValidator:
    """
    Validates optimizations against golden queries.
    
    Golden queries are carefully curated test cases that represent
    typical usage patterns and must pass validation for any optimizer.
    """
    
    def __init__(self, golden_queries_path: Optional[str] = None):
        self.golden_queries = self._load_golden_queries(golden_queries_path)
        self.baseline_metrics = {}
        
    def _load_golden_queries(self, path: Optional[str]) -> List[Dict]:
        """
        Load golden queries from file or use defaults.
        
        Args:
            path: Optional path to golden queries JSON file
            
        Returns:
            List of golden query test cases
        """
        if path:
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load golden queries from {path}: {e}")
        
        # Default golden queries for quick validation
        return [
            {
                "query": "debug payment processing error",
                "context": [
                    "Payment gateway integration failed with timeout error",
                    "User authentication successful",
                    "Database connection established",
                    "Payment processing started at 2024-12-12 10:30:00",
                    "Timeout occurred after 30 seconds waiting for payment response",
                    "Payment gateway returned error code 504",
                    "User session remains valid",
                    "Transaction rolled back successfully"
                ],
                "expected_relevance": 0.8,
                "min_pieces_retained": 3
            },
            {
                "query": "explain machine learning algorithms",
                "context": [
                    "Machine learning is a subset of artificial intelligence",
                    "Linear regression is used for predicting continuous values",
                    "Decision trees split data based on feature values",
                    "Random forests combine multiple decision trees",
                    "Neural networks are inspired by biological neurons",
                    "Support vector machines find optimal decision boundaries",
                    "K-means clustering groups similar data points",
                    "Deep learning uses multiple hidden layers"
                ],
                "expected_relevance": 0.9,
                "min_pieces_retained": 4
            },
            {
                "query": "how to implement authentication",
                "context": [
                    "JWT tokens provide stateless authentication",
                    "OAuth 2.0 is an authorization framework",
                    "Password hashing should use bcrypt or similar",
                    "Session management requires secure storage",
                    "Two-factor authentication adds security layer",
                    "API keys should be rotated regularly",
                    "HTTPS is mandatory for authentication endpoints",
                    "Rate limiting prevents brute force attacks"
                ],
                "expected_relevance": 0.85,
                "min_pieces_retained": 4
            }
        ]
    
    def validate_optimization(self, optimizer, result: Dict, 
                             query: str, original_context: List[str]) -> Dict:
        """
        Validate optimization result against quality criteria.
        
        Args:
            optimizer: The optimizer instance used
            result: Optimization result to validate
            query: Original query
            original_context: Original context pool
            
        Returns:
            Validation result with pass/fail status and details
        """
        validation_result = {
            'passed': True,
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        # Check 1: Token reduction achieved
        token_reduction = result['stats']['reduction']['token_reduction_pct']
        validation_result['checks']['token_reduction'] = {
            'value': token_reduction,
            'threshold': 0.15,
            'passed': token_reduction >= 0.15
        }
        
        if token_reduction < 0.15:
            validation_result['errors'].append(
                f"Token reduction {token_reduction:.1%} below minimum 15%"
            )
            validation_result['passed'] = False
        
        # Check 2: Processing time reasonable
        processing_time = result['stats']['processing_time_ms']
        validation_result['checks']['processing_time'] = {
            'value': processing_time,
            'threshold': 1000,  # 1 second
            'passed': processing_time <= 1000
        }
        
        if processing_time > 1000:
            validation_result['warnings'].append(
                f"Processing time {processing_time:.0f}ms exceeds target 1000ms"
            )
        
        # Check 3: No critical content lost (basic check)
        original_pieces = result['stats']['original']['pieces']
        final_pieces = result['stats']['final']['pieces']
        retention_ratio = final_pieces / original_pieces if original_pieces > 0 else 0
        
        validation_result['checks']['content_retention'] = {
            'value': retention_ratio,
            'threshold': 0.1,
            'passed': retention_ratio >= 0.1
        }
        
        if retention_ratio < 0.1:
            validation_result['errors'].append(
                f"Too aggressive reduction: only {retention_ratio:.1%} content retained"
            )
            validation_result['passed'] = False
        
        # Check 4: Result structure valid
        required_keys = ['optimized_context', 'stats', 'metadata']
        structure_valid = all(key in result for key in required_keys)
        
        validation_result['checks']['result_structure'] = {
            'required_keys': required_keys,
            'passed': structure_valid
        }
        
        if not structure_valid:
            validation_result['errors'].append("Result missing required keys")
            validation_result['passed'] = False
        
        return validation_result
    
    def run_golden_queries(self, optimizer) -> Dict:
        """
        Run all golden queries against optimizer.
        
        Args:
            optimizer: Optimizer instance to test
            
        Returns:
            Comprehensive validation results
        """
        results = {
            'total_queries': len(self.golden_queries),
            'passed': 0,
            'failed': 0,
            'details': [],
            'overall_passed': True
        }
        
        for i, golden_query in enumerate(self.golden_queries):
            query = golden_query['query']
            context = golden_query['context']
            
            # Run optimization
            optimization_result = optimizer.optimize(query, context)
            
            # Validate result
            validation_result = self.validate_optimization(
                optimizer, optimization_result, query, context
            )
            
            # Check golden query specific requirements
            if 'expected_relevance' in golden_query:
                # This would require a relevance scorer in full implementation
                validation_result['warnings'].append(
                    "Relevance check not implemented in Phase 1"
                )
            
            if 'min_pieces_retained' in golden_query:
                min_pieces = golden_query['min_pieces_retained']
                actual_pieces = optimization_result['stats']['final']['pieces']
                
                if actual_pieces < min_pieces:
                    validation_result['errors'].append(
                        f"Only {actual_pieces} pieces retained, expected >= {min_pieces}"
                    )
                    validation_result['passed'] = False
            
            # Record results
            query_result = {
                'query_index': i,
                'query': query,
                'passed': validation_result['passed'],
                'optimization_result': optimization_result,
                'validation': validation_result
            }
            
            results['details'].append(query_result)
            
            if validation_result['passed']:
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['overall_passed'] = False
        
        return results
    
    def generate_report(self, validation_results: Dict) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            validation_results: Results from run_golden_queries
            
        Returns:
            Formatted validation report
        """
        report = []
        report.append("CEGO Golden Query Validation Report")
        report.append("=" * 40)
        
        summary = validation_results
        report.append(f"Total Queries: {summary['total_queries']}")
        report.append(f"Passed: {summary['passed']}")
        report.append(f"Failed: {summary['failed']}")
        report.append(f"Success Rate: {summary['passed']/summary['total_queries']:.1%}")
        report.append(f"Overall: {'PASS' if summary['overall_passed'] else 'FAIL'}")
        report.append("")
        
        # Detail each query
        for detail in summary['details']:
            report.append(f"Query {detail['query_index'] + 1}: {detail['query'][:50]}...")
            report.append(f"  Status: {'PASS' if detail['passed'] else 'FAIL'}")
            
            opt_stats = detail['optimization_result']['stats']
            report.append(f"  Token Reduction: {opt_stats['reduction']['token_reduction_pct']:.1%}")
            report.append(f"  Processing Time: {opt_stats['processing_time_ms']:.0f}ms")
            
            if detail['validation']['errors']:
                report.append("  Errors:")
                for error in detail['validation']['errors']:
                    report.append(f"    - {error}")
            
            if detail['validation']['warnings']:
                report.append("  Warnings:")
                for warning in detail['validation']['warnings']:
                    report.append(f"    - {warning}")
            
            report.append("")
        
        return "\n".join(report)