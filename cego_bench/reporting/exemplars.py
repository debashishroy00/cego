"""Exemplar generation and side-by-side diff visualization for patent evidence."""

import difflib
import html
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..runners.metrics import BenchmarkMetrics
from ..runners.loaders import TestCase, load_jsonl_dataset


@dataclass
class ExemplarResult:
    """Result from optimizer for exemplar generation."""
    test_id: str
    optimizer: str
    reduction: float
    retention: float
    confidence: float
    latency_ms: float
    original_items: List[str]
    optimized_items: List[str]
    kept_indices: List[int]


class ExemplarGenerator:
    """Generate representative exemplars with side-by-side diffs."""

    def __init__(self, output_dir: Path):
        """Initialize exemplar generator.

        Args:
            output_dir: Directory for saving exemplar files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_exemplars(self,
                          metrics_list: List[BenchmarkMetrics],
                          test_cases: List[TestCase],
                          domain_mapping: Dict[str, str],
                          top_k: int = 3) -> Dict[str, List[Path]]:
        """Generate exemplar diffs for top performing cases per domain.

        Args:
            metrics_list: List of benchmark metrics
            test_cases: List of original test cases
            domain_mapping: Test ID to domain mapping
            top_k: Number of exemplars per domain

        Returns:
            Dictionary mapping domain to list of exemplar file paths
        """
        # Create test case lookup
        test_case_lookup = {tc.id: tc for tc in test_cases}

        # Group metrics by domain and optimizer
        domain_metrics = {}
        for metric in metrics_list:
            if not metric.success:
                continue

            domain = domain_mapping.get(metric.test_id, 'unknown')
            if domain not in domain_metrics:
                domain_metrics[domain] = {'pattern': [], 'entropy': []}

            domain_metrics[domain][metric.optimizer.value].append(metric)

        # Generate exemplars for each domain
        exemplar_files = {}

        for domain, optimizers in domain_metrics.items():
            print(f"Generating exemplars for domain: {domain}")

            # Select top k exemplars based on reduction while maintaining retention
            pattern_exemplars = self._select_top_exemplars(
                optimizers.get('pattern', []), test_case_lookup, top_k,
                min_retention=0.90
            )

            entropy_exemplars = self._select_top_exemplars(
                optimizers.get('entropy', []), test_case_lookup, top_k,
                min_retention=0.92
            )

            # Generate side-by-side diffs
            domain_files = []
            for i, (pattern_ex, entropy_ex) in enumerate(zip(pattern_exemplars, entropy_exemplars), 1):
                if pattern_ex and entropy_ex:
                    exemplar_file = self._create_exemplar_diff(
                        domain, i, pattern_ex, entropy_ex, test_case_lookup
                    )
                    domain_files.append(exemplar_file)

            exemplar_files[domain] = domain_files

        return exemplar_files

    def _select_top_exemplars(self,
                             metrics: List[BenchmarkMetrics],
                             test_case_lookup: Dict[str, TestCase],
                             top_k: int,
                             min_retention: float) -> List[Optional[BenchmarkMetrics]]:
        """Select top exemplars based on reduction while maintaining retention threshold.

        Args:
            metrics: List of metrics for this optimizer
            test_case_lookup: Test case lookup dictionary
            top_k: Number of exemplars to select
            min_retention: Minimum retention threshold

        Returns:
            List of top exemplars (may contain None if insufficient data)
        """
        # Filter by retention threshold
        qualified_metrics = [
            m for m in metrics
            if m.semantic_retention and m.semantic_retention >= min_retention
            and m.token_reduction_percentage is not None
        ]

        # Sort by reduction (descending)
        qualified_metrics.sort(key=lambda x: x.token_reduction_percentage, reverse=True)

        # Take top k, pad with None if needed
        exemplars = qualified_metrics[:top_k]
        while len(exemplars) < top_k:
            exemplars.append(None)

        return exemplars

    def _create_exemplar_diff(self,
                             domain: str,
                             exemplar_number: int,
                             pattern_metric: BenchmarkMetrics,
                             entropy_metric: BenchmarkMetrics,
                             test_case_lookup: Dict[str, TestCase]) -> Path:
        """Create side-by-side diff HTML for an exemplar.

        Args:
            domain: Domain name
            exemplar_number: Exemplar number within domain
            pattern_metric: Pattern optimizer metrics
            entropy_metric: Entropy optimizer metrics
            test_case_lookup: Test case lookup

        Returns:
            Path to generated HTML file
        """
        # Get original test case
        test_case = test_case_lookup.get(pattern_metric.test_id)
        if not test_case:
            raise ValueError(f"Test case not found: {pattern_metric.test_id}")

        # Prepare data for template
        original_items = [item.text for item in test_case.items]

        # Get optimized items (reconstruct from kept indices)
        pattern_optimized = [original_items[i] for i in pattern_metric.kept_indices
                           if i < len(original_items)]
        entropy_optimized = [original_items[i] for i in entropy_metric.kept_indices
                           if i < len(original_items)]

        # Generate HTML
        html_content = self._generate_exemplar_html(
            domain, exemplar_number, test_case, pattern_metric, entropy_metric,
            original_items, pattern_optimized, entropy_optimized
        )

        # Save file
        filename = f"{domain}_exemplar_{exemplar_number}_{test_case.id}.html"
        output_file = self.output_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_file

    def _generate_exemplar_html(self,
                               domain: str,
                               exemplar_number: int,
                               test_case: TestCase,
                               pattern_metric: BenchmarkMetrics,
                               entropy_metric: BenchmarkMetrics,
                               original_items: List[str],
                               pattern_optimized: List[str],
                               entropy_optimized: List[str]) -> str:
        """Generate HTML content for exemplar diff.

        Args:
            domain: Domain name
            exemplar_number: Exemplar number
            test_case: Original test case
            pattern_metric: Pattern optimizer metrics
            entropy_metric: Entropy optimizer metrics
            original_items: Original items list
            pattern_optimized: Pattern optimized items
            entropy_optimized: Entropy optimized items

        Returns:
            HTML content string
        """
        # Create diff tables
        pattern_diff_table = self._create_diff_table(original_items, pattern_optimized, "Pattern")
        entropy_diff_table = self._create_diff_table(original_items, entropy_optimized, "Entropy")

        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CEGO Exemplar: {domain} #{exemplar_number}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}

        .header .subtitle {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .metrics-section {{
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .metric-card.pattern {{
            border-left-color: #3498db;
        }}

        .metric-card.entropy {{
            border-left-color: #e74c3c;
        }}

        .metric-card h3 {{
            margin: 0 0 15px 0;
            color: #333;
            font-size: 1.3em;
        }}

        .metric-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 8px 0;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}

        .metric-item:last-child {{
            border-bottom: none;
        }}

        .metric-label {{
            font-weight: 500;
            color: #666;
        }}

        .metric-value {{
            font-weight: bold;
            color: #333;
            font-size: 1.1em;
        }}

        .diff-section {{
            padding: 30px;
        }}

        .diff-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}

        .diff-panel {{
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}

        .diff-header {{
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #ddd;
            font-weight: bold;
            font-size: 1.1em;
        }}

        .diff-header.pattern {{
            background: #e3f2fd;
            color: #1976d2;
        }}

        .diff-header.entropy {{
            background: #ffebee;
            color: #d32f2f;
        }}

        .diff-content {{
            max-height: 400px;
            overflow-y: auto;
        }}

        .diff-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .diff-table td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
            vertical-align: top;
        }}

        .item-kept {{
            background: #d4edda;
            border-left: 4px solid #28a745;
        }}

        .item-removed {{
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            opacity: 0.7;
        }}

        .item-number {{
            font-weight: bold;
            color: #666;
            width: 40px;
            text-align: center;
        }}

        .query-section {{
            background: #fff3cd;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
        }}

        .query-section h3 {{
            margin: 0 0 10px 0;
            color: #856404;
        }}

        .query-text {{
            font-style: italic;
            font-size: 1.1em;
            color: #333;
        }}

        .stats-summary {{
            background: #e8f5e8;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}

        .stats-summary h3 {{
            margin: 0 0 15px 0;
            color: #2e7d32;
        }}

        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .comparison-table th,
        .comparison-table td {{
            padding: 10px;
            border: 1px solid #ddd;
            text-align: center;
        }}

        .comparison-table th {{
            background: #f8f9fa;
            font-weight: bold;
        }}

        .better {{
            background: #d4edda;
            font-weight: bold;
        }}

        .worse {{
            background: #f8d7da;
        }}

        @media (max-width: 768px) {{
            .diff-container {{
                grid-template-columns: 1fr;
            }}

            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>CEGO Patent Exemplar</h1>
            <div class="subtitle">Domain: {domain.title()} | Exemplar #{exemplar_number} | Test ID: {test_case.id}</div>
        </div>

        <div class="query-section">
            <h3>Original Query</h3>
            <div class="query-text">"{html.escape(test_case.query)}"</div>
        </div>

        <div class="metrics-section">
            <div class="metrics-grid">
                <div class="metric-card pattern">
                    <h3>Pattern Optimizer</h3>
                    <div class="metric-item">
                        <span class="metric-label">Token Reduction:</span>
                        <span class="metric-value">{pattern_metric.token_reduction_percentage*100:.1f}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Semantic Retention:</span>
                        <span class="metric-value">{pattern_metric.semantic_retention:.3f}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Confidence:</span>
                        <span class="metric-value">{pattern_metric.confidence:.3f}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Latency:</span>
                        <span class="metric-value">{pattern_metric.latency_ms:.1f}ms</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Items Kept:</span>
                        <span class="metric-value">{len(pattern_optimized)}/{len(original_items)}</span>
                    </div>
                </div>

                <div class="metric-card entropy">
                    <h3>Entropy Optimizer</h3>
                    <div class="metric-item">
                        <span class="metric-label">Token Reduction:</span>
                        <span class="metric-value">{entropy_metric.token_reduction_percentage*100:.1f}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Semantic Retention:</span>
                        <span class="metric-value">{entropy_metric.semantic_retention:.3f}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Confidence:</span>
                        <span class="metric-value">{entropy_metric.confidence:.3f}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Latency:</span>
                        <span class="metric-value">{entropy_metric.latency_ms:.1f}ms</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Items Kept:</span>
                        <span class="metric-value">{len(entropy_optimized)}/{len(original_items)}</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="diff-section">
            <h2>Optimization Comparison</h2>

            <div class="diff-container">
                <div class="diff-panel">
                    <div class="diff-header pattern">Pattern Optimizer Results</div>
                    <div class="diff-content">
                        {pattern_diff_table}
                    </div>
                </div>

                <div class="diff-panel">
                    <div class="diff-header entropy">Entropy Optimizer Results</div>
                    <div class="diff-content">
                        {entropy_diff_table}
                    </div>
                </div>
            </div>

            <div class="stats-summary">
                <h3>Performance Comparison</h3>
                <table class="comparison-table">
                    <tr>
                        <th>Metric</th>
                        <th>Pattern</th>
                        <th>Entropy</th>
                        <th>Better</th>
                    </tr>
                    <tr>
                        <td>Token Reduction</td>
                        <td class="{'better' if (pattern_metric.token_reduction_percentage or 0) > (entropy_metric.token_reduction_percentage or 0) else ''}">{(pattern_metric.token_reduction_percentage or 0)*100:.1f}%</td>
                        <td class="{'better' if (entropy_metric.token_reduction_percentage or 0) > (pattern_metric.token_reduction_percentage or 0) else ''}">{(entropy_metric.token_reduction_percentage or 0)*100:.1f}%</td>
                        <td>{'Pattern' if (pattern_metric.token_reduction_percentage or 0) > (entropy_metric.token_reduction_percentage or 0) else 'Entropy'}</td>
                    </tr>
                    <tr>
                        <td>Semantic Retention</td>
                        <td class="{'better' if (pattern_metric.semantic_retention or 0) > (entropy_metric.semantic_retention or 0) else ''}">{pattern_metric.semantic_retention or 0:.3f}</td>
                        <td class="{'better' if (entropy_metric.semantic_retention or 0) > (pattern_metric.semantic_retention or 0) else ''}">{entropy_metric.semantic_retention or 0:.3f}</td>
                        <td>{'Pattern' if (pattern_metric.semantic_retention or 0) > (entropy_metric.semantic_retention or 0) else 'Entropy'}</td>
                    </tr>
                    <tr>
                        <td>Latency</td>
                        <td class="{'better' if (pattern_metric.latency_ms or 0) < (entropy_metric.latency_ms or 0) else ''}">{pattern_metric.latency_ms or 0:.1f}ms</td>
                        <td class="{'better' if (entropy_metric.latency_ms or 0) < (pattern_metric.latency_ms or 0) else ''}">{entropy_metric.latency_ms or 0:.1f}ms</td>
                        <td>{'Pattern' if (pattern_metric.latency_ms or 0) < (entropy_metric.latency_ms or 0) else 'Entropy'}</td>
                    </tr>
                </table>

                <p><strong>Key Insights:</strong></p>
                <ul>
                    <li>Entropy optimizer achieved {(entropy_metric.token_reduction_percentage or 0) - (pattern_metric.token_reduction_percentage or 0):+.1f}% more reduction than Pattern</li>
                    <li>Pattern optimizer maintained {(pattern_metric.semantic_retention or 0) - (entropy_metric.semantic_retention or 0):+.3f} higher retention</li>
                    <li>{'Pattern' if (pattern_metric.latency_ms or 0) < (entropy_metric.latency_ms or 0) else 'Entropy'} optimizer was {abs((pattern_metric.latency_ms or 0) - (entropy_metric.latency_ms or 0)):.1f}ms faster</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
        """

        return html_content

    def _create_diff_table(self, original_items: List[str], optimized_items: List[str], optimizer_name: str) -> str:
        """Create HTML diff table showing kept vs removed items.

        Args:
            original_items: Original list of items
            optimized_items: Optimized list of items
            optimizer_name: Name of optimizer

        Returns:
            HTML table string
        """
        kept_items = set(optimized_items)

        table_rows = []
        for i, item in enumerate(original_items):
            is_kept = item in kept_items
            css_class = "item-kept" if is_kept else "item-removed"
            status = "KEPT" if is_kept else "REMOVED"

            # Truncate long items for display
            display_text = item[:200] + "..." if len(item) > 200 else item

            row = f"""
            <tr class="{css_class}">
                <td class="item-number">{i+1}</td>
                <td>{html.escape(display_text)}</td>
                <td style="font-weight: bold; text-align: center;">{status}</td>
            </tr>
            """
            table_rows.append(row)

        table_html = f"""
        <table class="diff-table">
            <thead>
                <tr style="background: #f8f9fa;">
                    <th class="item-number">#</th>
                    <th>Item Text</th>
                    <th style="text-align: center;">Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        """

        return table_html

    def create_domain_summary(self, domain_exemplars: Dict[str, List[Path]]) -> Path:
        """Create summary HTML linking all exemplars by domain.

        Args:
            domain_exemplars: Dictionary mapping domains to exemplar file paths

        Returns:
            Path to summary HTML file
        """
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CEGO Exemplars Summary</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}

        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}

        .domain-section {{
            margin-bottom: 40px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}

        .domain-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            font-size: 1.5em;
            font-weight: bold;
        }}

        .exemplar-list {{
            padding: 20px;
        }}

        .exemplar-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            margin: 10px 0;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}

        .exemplar-item:hover {{
            background: #e9ecef;
            transform: translateY(-1px);
            transition: all 0.2s;
        }}

        .exemplar-title {{
            font-weight: bold;
            color: #333;
        }}

        .exemplar-link {{
            background: #667eea;
            color: white;
            padding: 8px 16px;
            text-decoration: none;
            border-radius: 4px;
            font-weight: bold;
        }}

        .exemplar-link:hover {{
            background: #5a6fd8;
        }}

        .stats {{
            text-align: center;
            background: #e8f5e8;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}

        .stats h3 {{
            margin: 0 0 10px 0;
            color: #2e7d32;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CEGO Patent Exemplars Summary</h1>

        <div class="stats">
            <h3>Summary Statistics</h3>
            <p>Total Domains: {len(domain_exemplars)} | Total Exemplars: {sum(len(files) for files in domain_exemplars.values())}</p>
        </div>
        """

        for domain, exemplar_files in domain_exemplars.items():
            html_content += f"""
        <div class="domain-section">
            <div class="domain-header">
                {domain.title()} Domain ({len(exemplar_files)} exemplars)
            </div>
            <div class="exemplar-list">
            """

            for i, exemplar_file in enumerate(exemplar_files, 1):
                filename = exemplar_file.name
                html_content += f"""
                <div class="exemplar-item">
                    <div class="exemplar-title">Exemplar #{i}</div>
                    <a href="{filename}" class="exemplar-link">View Details</a>
                </div>
                """

            html_content += """
            </div>
        </div>
            """

        html_content += """
    </div>
</body>
</html>
        """

        # Save summary file
        summary_file = self.output_dir / "exemplars_summary.html"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return summary_file