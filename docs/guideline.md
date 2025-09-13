CEGO Development Guidelines for Claude Code
Project Context
You are implementing CEGO (Context Entropy Gradient Optimization) - a system that reduces LLM token usage by 60-80% while improving accuracy. The project has a provisional patent pending and follows a phased implementation approach.
MANDATORY CODING STANDARDS
1. File Size Limits

Target: 300 lines per file
Hard limit: 500 lines (commits blocked above this)
Function limit: 30 lines per function
Class limit: 150 lines per class
If approaching limits, refactor BEFORE adding features

2. Project Structure
cego/
├── src/
│   ├── core/           # Core algorithms (entropy, gradient)
│   ├── optimizers/     # Different optimization strategies
│   ├── validators/     # Validation and testing
│   ├── utils/         # Shared utilities
│   └── api/           # External interfaces
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── configs/
└── docs/
3. Code Organization Rules

Single Responsibility: Each file/class/function does ONE thing
No God Objects: Split large classes into composable components
Clear Separation: Business logic separate from I/O operations
Explicit Dependencies: No hidden imports or global state

CEGO-SPECIFIC PATTERNS
1. Algorithm Implementation Pattern
pythonclass BaseOptimizer(ABC):
    """All optimizers follow this pattern"""
    
    @abstractmethod
    def calculate_entropy(self, context: List[str]) -> float:
        """Pure function - no side effects"""
        pass
    
    @abstractmethod
    def compute_gradient(self, context: List[str]) -> np.ndarray:
        """Stateless computation"""
        pass
    
    def optimize(self, query: str, context_pool: List[str]) -> Dict:
        """Main entry point - orchestrates the algorithm"""
        # 1. Validate inputs
        # 2. Apply optimization
        # 3. Track metrics
        # 4. Return results with stats
        pass
2. Metrics & Monitoring Pattern
Every optimization MUST return:
python{
    'optimized_context': List[str],
    'stats': {
        'original': {'pieces': int, 'tokens': int},
        'final': {'pieces': int, 'tokens': int},
        'reduction': {
            'token_reduction_pct': float,
            'pieces_saved': int
        },
        'processing_time_ms': float,
        'algorithm_used': str,
        'phase_transitions': List[dict]
    },
    'metadata': {
        'version': str,
        'timestamp': str,
        'rollback_available': bool
    }
}
3. Validation Pattern
pythonclass Validator:
    def __init__(self, golden_queries: List[Dict]):
        self.golden_queries = golden_queries
        self.baseline_metrics = {}
    
    def validate_optimization(self, result: Dict) -> bool:
        """Returns True if optimization meets quality bar"""
        # Check: token reduction >= 15%
        # Check: relevance score >= 0.85
        # Check: no critical content lost
        pass
DEVELOPMENT PHASES & CHECKPOINTS
Phase 1: Quick Wins (Week 1)

Focus: quick_wins.py implementation
Target: 20-30% reduction
Validation: Must pass all golden queries

Phase 2: MVP (Week 2-4)

Focus: Basic entropy calculation
Target: 30-45% reduction
Add: Gradient computation, simple phase detection

Phase 3: Advanced (Week 5+)

Focus: Full CEGO algorithm
Target: 60-80% reduction
Add: Adaptive entropy, dynamic lambda, ensemble methods

BEFORE WRITING ANY CODE
Ask yourself:

Does similar functionality already exist in the codebase?
Which files will I modify? Are any already >250 lines?
Can this be broken into smaller, testable functions?
Will this work with the rollback mechanism?
Have I checked the existing patterns in quick_wins.py?

QUALITY CHECKLIST (Run After Each Feature)
bash# Check file sizes
find src -name "*.py" | xargs wc -l | sort -rn | head -10

# Run tests
pytest tests/ -v

# Check complexity
radon cc src/ -nb

# Validate performance
python benchmarks/validate_reduction.py

# Ensure rollback works
python tests/test_rollback.py
TESTING REQUIREMENTS
Every new optimizer MUST have:

Unit tests for each method
Integration test with real data
Performance benchmark
Rollback test
Golden query validation

COMMIT MESSAGE FORMAT
[PHASE] Component: Brief description

- Specific change 1
- Specific change 2

Metrics: XX% reduction, XXms latency
Tests: All passing
Example:
[MVP] Entropy: Add adaptive entropy calculation

- Implement Shannon/cross-entropy selection based on skewness
- Add thresholds: 0.5 for low, 2.0 for high skewness

Metrics: 35% reduction, 450ms latency
Tests: All passing (including golden queries)
ERROR HANDLING PATTERN
pythondef optimize_with_fallback(self, query: str, context: List[str]) -> Dict:
    try:
        result = self.optimize(query, context)
        if self.validate(result):
            return result
    except Exception as e:
        logger.warning(f"Optimization failed: {e}")
    
    # Always have a fallback
    return self.quick_wins_fallback(query, context)
PERFORMANCE STANDARDS

No function should take >500ms for 1000 contexts
Memory usage should stay under 500MB for typical workloads
Always profile before optimizing
Use generators for large datasets

DOCUMENTATION REQUIREMENTS
Every public function needs:
pythondef function_name(param1: Type, param2: Type) -> ReturnType:
    """
    Brief description (one line).
    
    Detailed explanation if needed.
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When this happens
        
    Example:
        >>> result = function_name(x, y)
        >>> print(result['stats']['reduction'])
        0.35
    """
INTEGRATION POINTS
When implementing new features, ensure compatibility with:

Rollback System: Every optimization must be reversible
Monitoring: All operations must emit metrics
Validation: Must work with golden query framework
Caching: Should leverage existing cache when available
API: Must follow the standard input/output format

PATENT CONSIDERATIONS

Keep implementation details modular (supports patent claims)
Document any novel optimizations discovered
Track performance metrics for patent updates
Maintain clear separation between algorithm and implementation

SESSION START CHECKLIST
When CC starts a new session, always:

Review this guideline document
Check current file sizes: find src -name "*.py" | xargs wc -l | sort -rn | head -5
Run tests to ensure nothing is broken: pytest tests/ -q
Review the latest metrics in benchmarks/results.json
Check which phase we're in and current targets

RED FLAGS - Stop and Refactor If:

Any file exceeds 400 lines
Token reduction drops below 15%
Processing time exceeds 1 second for standard workload
Test coverage drops below 80%
Rollback mechanism fails
Golden queries fail validation

REMEMBER
The goal is "Ambitious Vision + Conservative Execution". We're building a production system that will handle millions of requests. Every optimization must be:

Reliable: Works every time
Reversible: Can rollback if needed
Measurable: Metrics for everything
Maintainable: Clear, simple code


Use this guideline in every Claude Code session. Start by confirming you've read and will follow these standards.