# AI-Assisted Development Pipeline v2.0
*A lightweight, project-agnostic orchestrator for production-ready code*

## Quick Start
Give this template to Claude Code or your preferred AI assistant to scaffold an automated development pipeline that maintains investor-grade quality while enabling rapid experimentation.

## Core Philosophy
- **One lane, clear gates**: Simple linear workflow with automated quality checks
- **Language agnostic**: Adapts to Python, TypeScript, Go, or any ecosystem
- **Token efficient**: Agents work with diffs and summaries, not entire codebases
- **Production ready**: Every change meets security, performance, and quality standards
- **Developer friendly**: Augments, doesn't replace, human creativity

## The Pipeline

```
TASK → PLAN → BUILD → TEST → REVIEW → SECURITY → DECIDE → DOCS → REPORT
         ↑                                              ↓
         └──────────── RETRY (max 2) ──────────────────┘
```

## Project Standard (PROJECT_STD v2.0)

### Core Rules
1. **Simplicity First**: Functions ≤30 lines, single responsibility
2. **Test Everything**: Every change needs a test (unit/integration)
3. **Measure Impact**: Track performance metrics for every change
4. **Security Always**: No secrets, no unsafe operations, scan everything
5. **Document Decisions**: Auto-generate changelog and architecture notes

### Quality Gates
- **BUILD**: Code compiles/interprets without errors
- **TEST**: All tests pass, including new ones
- **REVIEW**: Meets style guide, no dead code, follows patterns
- **SECURITY**: No high-severity issues, no exposed secrets
- **BENCH**: Performance within acceptable bounds

## Implementation Structure

### 1. Orchestrator Core
```typescript
// /orchestrator/index.ts (or .py for Python projects)
interface TaskSpec {
  task: string                    // What to build
  targets: string[]               // Files to modify
  acceptance: {
    tests: string[]              // Required test files
    bench: {                     // Performance bounds
      targets: Record<string,number>  // Min improvements
      caps?: Record<string,number>    // Max regressions
    }
    security: "no-high" | "no-medium"
  }
}

// State machine drives the pipeline
async function orchestrate(spec: TaskSpec) {
  // 1. Builder creates patch
  // 2. Run tests/bench/security
  // 3. Review checks quality
  // 4. Security scans risks
  // 5. Decide PROMOTE/HOLD
  // 6. Generate docs if PROMOTE
  // 7. Report results
}
```

### 2. Agent Prompts

**/agents/builder.prompt**
```
You are a Builder agent. Your task: {TASK}
Target files: {FILES}
Rules:
- Functions ≤30 lines
- Include unit tests
- Follow existing patterns in codebase
- Output only the patch/diff
Context: {CURRENT_CODE}
```

**/agents/reviewer.prompt**
```
You are a Reviewer agent. Check this patch:
{PATCH}
Return PASS/FAIL for:
□ Follows code style
□ Has tests
□ No dead code
□ Functions ≤30 lines
□ Uses existing patterns
```

**/agents/security.prompt**
```
You are a Security agent. Scan for:
- Hardcoded secrets/keys
- SQL/command injection risks
- Path traversal vulnerabilities
- Unsafe regex patterns
- Exposed sensitive data
Return: CLEAR or list of issues with severity
```

### 3. CI Integration

**For Python Projects (pytest, ruff, bandit)**
```yaml
# .github/workflows/ci.yml
name: Pipeline
on: [push, pull_request]
jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -r requirements-dev.txt
      - run: ruff check . || echo "LINT: errors=$(ruff check . --count)" > ci/LINT_SUMMARY.txt
      - run: pytest --cov || echo "TEST: $(pytest -q --tb=no | tail -1)" > ci/TEST_SUMMARY.txt
      - run: python bench/run.py || echo "BENCH: latency:+0% memory:+0%" > ci/BENCH_SUMMARY.txt
      - run: bandit -r . || echo "SEC: high=0 medium=0" > ci/SEC_SUMMARY.txt
```

**For TypeScript Projects (vitest, eslint, semgrep)**
```yaml
# .github/workflows/ci.yml
name: Pipeline
on: [push, pull_request]
jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm ci
      - run: npm run lint || echo "LINT: errors=$(npx eslint . --format compact | grep -c error)" > ci/LINT_SUMMARY.txt
      - run: npm test -- --coverage || echo "TEST: pass=X fail=Y" > ci/TEST_SUMMARY.txt
      - run: npm run bench || echo "BENCH: throughput:+0% latency:+0%" > ci/BENCH_SUMMARY.txt
      - run: semgrep --config auto || echo "SEC: high=0" > ci/SEC_SUMMARY.txt
```

### 4. Language-Specific Tooling

**Python Setup**
```toml
# pyproject.toml
[tool.ruff]
line-length = 100
select = ["E", "F", "B", "S", "I"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src --cov-report=term"

[tool.bandit]
exclude_dirs = ["tests", "venv"]
```

**TypeScript Setup**
```json
// package.json
{
  "scripts": {
    "lint": "eslint . --max-warnings 0",
    "test": "vitest --run",
    "bench": "node bench/run.js",
    "security": "semgrep --config auto && gitleaks detect"
  },
  "lint-staged": {
    "*.{ts,js}": ["eslint --fix", "prettier --write"]
  }
}
```

### 5. Metrics Configuration

**Define Your Metrics**
```javascript
// /config/metrics.js (or .py)
export const METRICS = {
  // Performance
  latency: { target: -5, cap: 10 },      // -5% improvement, max 10% regression
  memory: { target: 0, cap: 5 },         // No increase, max 5% regression
  throughput: { target: 0, cap: -10 },   // No decrease, max 10% reduction

  // Quality
  coverage: { target: 80, absolute: true }, // Minimum 80% coverage
  complexity: { target: 10, absolute: true }, // Max cyclomatic complexity

  // Custom for your domain (e.g., CEGO)
  optimization_ratio: { target: 1.2, absolute: true }, // Custom metric
}
```

## Practical Usage

### Example 1: Adding a Feature
```bash
# Create task specification
cat > task.json << EOF
{
  "task": "Add caching to API endpoints",
  "targets": ["src/api/handlers.py", "tests/test_handlers.py"],
  "acceptance": {
    "tests": ["tests/test_handlers.py"],
    "bench": {
      "targets": { "latency": -20, "memory": 0 },
      "caps": { "memory": 10 }
    },
    "security": "no-high"
  }
}
EOF

# Run orchestrator
python orchestrator/run.py task.json

# Output:
# DECISION: PROMOTE
# - Latency: -23% ✓
# - Memory: +3% ✓
# - Tests: 15/15 passed ✓
# - Security: CLEAR ✓
# Next: Deploy behind feature flag
```

### Example 2: Optimization Task
```bash
# For CEGO-specific optimization
cat > task.json << EOF
{
  "task": "Optimize pattern recognition algorithm",
  "targets": ["backend/optimizers/pattern_recognition.py"],
  "acceptance": {
    "tests": ["tests/test_pattern_recognition.py"],
    "bench": {
      "targets": { "optimization_ratio": 1.5, "latency": -10 },
      "caps": { "memory": 20 }
    },
    "security": "no-medium"
  }
}
EOF
```

## Adapting to Your Project

### Step 1: Choose Your Language Tools
- **Python**: ruff, black, pytest, bandit, safety
- **TypeScript**: eslint, prettier, vitest, semgrep
- **Go**: gofmt, go test, gosec, staticcheck
- **Rust**: clippy, cargo test, cargo-audit

### Step 2: Define Your Metrics
What matters for your domain?
- **API service**: latency, throughput, error rate
- **ML/Optimization**: accuracy, convergence time, memory
- **Frontend**: bundle size, render time, lighthouse score

### Step 3: Set Your Standards
```yaml
# /.pipeline/standards.yml
code:
  max_function_lines: 30
  max_file_lines: 300
  test_coverage: 80

performance:
  latency_regression: 10%  # Max allowed
  memory_regression: 20%   # Max allowed

security:
  severity_threshold: "medium"
  secret_scanning: true
  dependency_scanning: true
```

### Step 4: Wire Your CI
1. Add the tooling to your CI config
2. Output summaries to `/ci/*.txt` files
3. Point orchestrator at these summaries

## Benefits for Stakeholders

### For Developers
- **Faster iteration**: Automated reviews catch issues early
- **Learn patterns**: AI agents enforce best practices consistently
- **Focus on creativity**: Spend time on algorithms, not formatting

### For Investors
- **Professional codebase**: Every commit meets production standards
- **Security built-in**: No embarrassing breaches or exposed keys
- **Measurable quality**: Metrics dashboard shows continuous improvement

### For Users
- **Reliable software**: Thoroughly tested, predictable performance
- **Fast features**: Automation enables rapid, safe deployment
- **Transparent changes**: Auto-generated docs explain every update

## Migration Path

### Week 1: Foundation
- [ ] Install linting/formatting tools
- [ ] Add pre-commit hooks
- [ ] Set up basic CI pipeline

### Week 2: Agents
- [ ] Create agent prompt templates
- [ ] Wire orchestrator to your LLM API
- [ ] Test with simple tasks

### Week 3: Metrics
- [ ] Define domain-specific benchmarks
- [ ] Integrate benchmark runner
- [ ] Set performance gates

### Week 4: Production
- [ ] Run full pipeline on feature branch
- [ ] Tune prompts based on results
- [ ] Document team workflows

## Advanced Features

### Multi-Language Support
```python
# orchestrator/adapters.py
class LanguageAdapter:
    def get_tools(self, language: str):
        return {
            "python": {"lint": "ruff", "test": "pytest", "security": "bandit"},
            "typescript": {"lint": "eslint", "test": "vitest", "security": "semgrep"},
            "go": {"lint": "gofmt", "test": "go test", "security": "gosec"}
        }[language]
```

### Custom Agents
```python
# agents/domain_expert.py
class DomainExpertAgent:
    """For CEGO: understands optimization algorithms"""
    def review_optimization(self, patch, metrics):
        # Check if optimization actually improves convergence
        # Verify mathematical correctness
        # Ensure benchmark comparisons are fair
        pass
```

### Intelligent Retry
```python
# orchestrator/retry_strategy.py
def smart_retry(failure_reason: str, attempt: int):
    if "timeout" in failure_reason and attempt < 2:
        return "increase_timeout"
    elif "test_failed" in failure_reason:
        return "fix_test_first"
    elif "performance_regression" in failure_reason:
        return "profile_and_optimize"
    return "ask_human"
```

## FAQ

**Q: What if I don't want AI to write my code?**
A: Use it for review only. The pipeline works with human-written code too.

**Q: Can this work with existing codebases?**
A: Yes. Start with new features, gradually add standards to old code.

**Q: What about complex architectural decisions?**
A: The orchestrator flags these for human review. It handles routine tasks.

**Q: How much does this cost in LLM tokens?**
A: ~$0.10-0.50 per feature with GPT-4, less with Claude Haiku or local models.

**Q: Can I customize the workflow?**
A: Yes. The state machine is configurable. Add/remove stages as needed.

## Next Steps

1. **Try it**: Copy this template, adapt to your language
2. **Measure it**: Track how much time it saves
3. **Improve it**: Tune prompts based on your domain
4. **Share it**: Contribute improvements back

---

*Remember: This pipeline augments human creativity, it doesn't replace it. Use it to handle the repetitive parts so you can focus on the innovative parts.*