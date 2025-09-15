"""HTML report generation for CEGO benchmark results."""

import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

from ..runners.metrics import BenchmarkMetrics
from .aggregate import aggregate_metrics, create_summary_table, create_calibration_table


def generate_html_report(metrics_list: List[BenchmarkMetrics],
                        output_file: Path,
                        run_config: Dict[str, Any],
                        domain_mapping: Optional[Dict[str, str]] = None,
                        acceptance_gates: Optional[Dict[str, Any]] = None,
                        suggested_tweaks: Optional[Dict[str, Any]] = None) -> None:
    """Generate comprehensive HTML report.

    Args:
        metrics_list: List of benchmark metrics
        output_file: Path to output HTML file
        run_config: Benchmark run configuration
        domain_mapping: Optional mapping from test_id to domain
        acceptance_gates: Optional acceptance gate results
        suggested_tweaks: Optional suggested parameter tweaks
    """
    # Aggregate data
    aggregated_stats = aggregate_metrics(metrics_list)
    summary_table = create_summary_table(aggregated_stats)
    calibration_table = create_calibration_table(metrics_list)

    # Generate domain breakdown if available
    domain_stats = {}
    if domain_mapping:
        from .aggregate import aggregate_by_domain
        domain_stats = aggregate_by_domain(metrics_list, domain_mapping)

    # Prepare template data
    template_data = {
        "title": "CEGO Benchmark Report",
        "generated_at": datetime.now().isoformat(),
        "config": run_config,
        "aggregated_stats": aggregated_stats,
        "summary_table_html": summary_table.to_html(classes="table table-striped", escape=False),
        "calibration_table_html": calibration_table.to_html(classes="table table-striped", escape=False),
        "domain_stats": domain_stats,
        "acceptance_gates": acceptance_gates,
        "suggested_tweaks": suggested_tweaks,
        "total_tests": len(metrics_list),
        "successful_tests": len([m for m in metrics_list if m.success]),
    }

    # Generate HTML content
    html_content = _generate_html_template(template_data)

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)


def _generate_html_template(data: Dict[str, Any]) -> str:
    """Generate HTML template with embedded CSS and JavaScript.

    Args:
        data: Template data dictionary

    Returns:
        Complete HTML string
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data['title']}</title>
    <style>
        {_get_embedded_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{data['title']}</h1>
            <p class="subtitle">Generated: {data['generated_at']}</p>
        </header>

        <div class="tabs">
            <button class="tab-button active" onclick="openTab(event, 'overview')">Overview</button>
            <button class="tab-button" onclick="openTab(event, 'domains')">By Domain</button>
            <button class="tab-button" onclick="openTab(event, 'calibration')">Calibration</button>
            <button class="tab-button" onclick="openTab(event, 'acceptance')">Gates & Tweaks</button>
            <button class="tab-button" onclick="openTab(event, 'config')">Configuration</button>
        </div>

        <div id="overview" class="tab-content active">
            {_generate_overview_section(data)}
        </div>

        <div id="domains" class="tab-content">
            {_generate_domains_section(data)}
        </div>

        <div id="calibration" class="tab-content">
            {_generate_calibration_section(data)}
        </div>

        <div id="acceptance" class="tab-content">
            {_generate_acceptance_section(data)}
        </div>

        <div id="config" class="tab-content">
            {_generate_config_section(data)}
        </div>
    </div>

    <script>
        {_get_embedded_javascript()}
    </script>
</body>
</html>"""


def _generate_overview_section(data: Dict[str, Any]) -> str:
    """Generate overview section HTML."""
    stats = data["aggregated_stats"]

    # Extract key metrics for display
    pattern_stats = stats.get("pattern", {})
    entropy_stats = stats.get("entropy", {})

    return f"""
        <section>
            <h2>Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card pattern">
                    <h3>Pattern Optimizer</h3>
                    <div class="metric">
                        <span class="value">{pattern_stats.get('reduction_median', 0)*100:.1f}%</span>
                        <span class="label">Median Reduction</span>
                    </div>
                    <div class="metric">
                        <span class="value">{pattern_stats.get('retention_mean', 0):.3f}</span>
                        <span class="label">Mean Retention</span>
                    </div>
                    <div class="metric">
                        <span class="value">{pattern_stats.get('latency_p90', 0):.1f}ms</span>
                        <span class="label">P90 Latency</span>
                    </div>
                </div>

                <div class="metric-card entropy">
                    <h3>Entropy Optimizer</h3>
                    <div class="metric">
                        <span class="value">{entropy_stats.get('reduction_median', 0)*100:.1f}%</span>
                        <span class="label">Median Reduction</span>
                    </div>
                    <div class="metric">
                        <span class="value">{entropy_stats.get('retention_mean', 0):.3f}</span>
                        <span class="label">Mean Retention</span>
                    </div>
                    <div class="metric">
                        <span class="value">{entropy_stats.get('latency_p90', 0):.1f}ms</span>
                        <span class="label">P90 Latency</span>
                    </div>
                </div>
            </div>

            <h3>Detailed Statistics</h3>
            {data['summary_table_html']}

            <h3>Junk Handling Performance</h3>
            <div class="junk-stats">
                <p><strong>Pattern Junk Kept Rate:</strong> {pattern_stats.get('junk_kept_rate_median', 'N/A')}</p>
                <p><strong>Entropy Junk Kept Rate:</strong> {entropy_stats.get('junk_kept_rate_median', 'N/A')}</p>
                <p><strong>Pattern Intent Preservation:</strong> {pattern_stats.get('intent_preservation_rate', 'N/A')}</p>
                <p><strong>Entropy Intent Preservation:</strong> {entropy_stats.get('intent_preservation_rate', 'N/A')}</p>
            </div>
        </section>
    """


def _generate_domains_section(data: Dict[str, Any]) -> str:
    """Generate domains section HTML."""
    domain_stats = data.get("domain_stats", {})

    if not domain_stats:
        return "<p>No domain breakdown available.</p>"

    content = "<h2>Performance by Domain</h2>"

    for domain, stats in domain_stats.items():
        pattern_stats = stats.get("pattern", {})
        entropy_stats = stats.get("entropy", {})

        # Format numeric values safely
        pattern_reduction = pattern_stats.get('reduction_median', 'N/A')
        pattern_retention = pattern_stats.get('retention_mean', 'N/A')
        pattern_success = pattern_stats.get('success_rate', 0) * 100

        entropy_reduction = entropy_stats.get('reduction_median', 'N/A')
        entropy_retention = entropy_stats.get('retention_mean', 'N/A')
        entropy_success = entropy_stats.get('success_rate', 0) * 100

        pattern_reduction_str = f"{pattern_reduction*100:.1f}%" if isinstance(pattern_reduction, (int, float)) else str(pattern_reduction)
        pattern_retention_str = f"{pattern_retention:.3f}" if isinstance(pattern_retention, (int, float)) else str(pattern_retention)

        entropy_reduction_str = f"{entropy_reduction*100:.1f}%" if isinstance(entropy_reduction, (int, float)) else str(entropy_reduction)
        entropy_retention_str = f"{entropy_retention:.3f}" if isinstance(entropy_retention, (int, float)) else str(entropy_retention)

        content += f"""
        <div class="domain-section">
            <h3>Domain: {domain}</h3>
            <div class="domain-grid">
                <div class="domain-card">
                    <h4>Pattern</h4>
                    <p>Reduction: {pattern_reduction_str}</p>
                    <p>Retention: {pattern_retention_str}</p>
                    <p>Success: {pattern_success:.1f}%</p>
                </div>
                <div class="domain-card">
                    <h4>Entropy</h4>
                    <p>Reduction: {entropy_reduction_str}</p>
                    <p>Retention: {entropy_retention_str}</p>
                    <p>Success: {entropy_success:.1f}%</p>
                </div>
            </div>
        </div>
        """

    return content


def _generate_calibration_section(data: Dict[str, Any]) -> str:
    """Generate calibration section HTML."""
    calibration_stats = data["aggregated_stats"].get("calibration", {})
    ece = calibration_stats.get("expected_calibration_error", 0)

    content = f"""
        <h2>Confidence Calibration Analysis</h2>
        <div class="calibration-summary">
            <p><strong>Expected Calibration Error (ECE):</strong> {ece:.4f}</p>
            <p class="calibration-note">
                Lower is better. ECE measures how well confidence scores align with actual performance.
            </p>
        </div>

        <h3>Calibration by Confidence Bins</h3>
        {data['calibration_table_html']}

        <div class="calibration-explanation">
            <h4>Interpretation Guide</h4>
            <ul>
                <li><strong>Well calibrated:</strong> Average confidence ≈ Average accuracy</li>
                <li><strong>Overconfident:</strong> Average confidence > Average accuracy</li>
                <li><strong>Underconfident:</strong> Average confidence < Average accuracy</li>
            </ul>
        </div>
    """

    return content


def _generate_acceptance_section(data: Dict[str, Any]) -> str:
    """Generate acceptance gates and tweaks section HTML."""
    gates = data.get("acceptance_gates", {})
    tweaks = data.get("suggested_tweaks", {})

    content = "<h2>Acceptance Gates & Suggested Tweaks</h2>"

    if gates:
        content += "<h3>Acceptance Gate Results</h3>"
        content += "<div class='gates-grid'>"

        for gate_name, result in gates.items():
            # Skip non-dict results like 'overall_passed'
            if not isinstance(result, dict):
                continue

            status_class = "pass" if result.get("passed", False) else "fail"
            content += f"""
            <div class="gate-card {status_class}">
                <h4>{gate_name}</h4>
                <p><strong>Status:</strong> {'PASS' if result.get('passed') else 'FAIL'}</p>
                <p><strong>Actual:</strong> {result.get('actual', 'N/A')}</p>
                <p><strong>Required:</strong> {result.get('required', 'N/A')}</p>
            </div>
            """

        content += "</div>"

    if tweaks:
        content += "<h3>Suggested Parameter Tweaks</h3>"
        content += "<div class='tweaks-section'>"
        content += "<p>The following parameter adjustments are suggested to meet acceptance criteria:</p>"
        content += "<pre class='tweaks-yaml'>"
        content += _format_tweaks_as_yaml(tweaks)
        content += "</pre>"
        content += "</div>"

    if not gates and not tweaks:
        content += "<p>No acceptance gate results or suggested tweaks available.</p>"

    return content


def _generate_config_section(data: Dict[str, Any]) -> str:
    """Generate configuration section HTML."""
    config = data.get("config", {})

    return f"""
        <h2>Benchmark Configuration</h2>
        <div class="config-section">
            <h3>Run Parameters</h3>
            <ul>
                <li><strong>Total Tests:</strong> {data.get('total_tests', 0)}</li>
                <li><strong>Successful Tests:</strong> {data.get('successful_tests', 0)}</li>
                <li><strong>Datasets:</strong> {len(config.get('datasets', []))}</li>
                <li><strong>Repeat Count:</strong> {config.get('runs', {}).get('repeat', 1)}</li>
            </ul>

            <h3>Full Configuration</h3>
            <pre class="config-json">{json.dumps(config, indent=2, default=str)}</pre>
        </div>
    """


def _format_tweaks_as_yaml(tweaks: Dict[str, Any]) -> str:
    """Format tweaks dictionary as YAML string."""
    lines = []
    for key, value in tweaks.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for subkey, subvalue in value.items():
                lines.append(f"  {subkey}: {subvalue}")
        else:
            lines.append(f"{key}: {value}")
    return "\\n".join(lines)


def _get_embedded_css() -> str:
    """Get embedded CSS styles."""
    return """
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            min-height: 100vh;
        }

        header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
        }

        h1 {
            color: #007acc;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #666;
            font-size: 1.1em;
        }

        .tabs {
            display: flex;
            border-bottom: 2px solid #ddd;
            margin-bottom: 20px;
        }

        .tab-button {
            background: none;
            border: none;
            padding: 12px 24px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            font-size: 1.1em;
            transition: all 0.3s;
        }

        .tab-button.active {
            border-bottom-color: #007acc;
            color: #007acc;
            font-weight: bold;
        }

        .tab-button:hover {
            background-color: #f0f0f0;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }

        .metric-card {
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .metric-card.pattern {
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
            border-left: 4px solid #2196f3;
        }

        .metric-card.entropy {
            background: linear-gradient(135deg, #f3e5f5, #e1bee7);
            border-left: 4px solid #9c27b0;
        }

        .metric-card h3 {
            margin-bottom: 15px;
            font-size: 1.3em;
        }

        .metric {
            margin-bottom: 10px;
        }

        .metric .value {
            display: block;
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }

        .metric .label {
            font-size: 0.9em;
            color: #666;
        }

        .table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        .table th,
        .table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        .table th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #333;
        }

        .table-striped tbody tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        .junk-stats {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }

        .domain-section {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
        }

        .domain-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 15px;
        }

        .domain-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
        }

        .calibration-summary {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #2196f3;
        }

        .calibration-note {
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }

        .calibration-explanation {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }

        .gates-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .gate-card {
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid;
        }

        .gate-card.pass {
            background: #e8f5e8;
            border-left-color: #4caf50;
        }

        .gate-card.fail {
            background: #ffebee;
            border-left-color: #f44336;
        }

        .tweaks-section {
            margin: 20px 0;
        }

        .tweaks-yaml {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border: 1px solid #ddd;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }

        .config-json {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border: 1px solid #ddd;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
            max-height: 400px;
            overflow-y: auto;
        }

        @media (max-width: 768px) {
            .metrics-grid,
            .domain-grid,
            .gates-grid {
                grid-template-columns: 1fr;
            }

            .tabs {
                flex-wrap: wrap;
            }

            .tab-button {
                flex: 1;
                min-width: 120px;
            }
        }
    """


def _get_embedded_javascript() -> str:
    """Get embedded JavaScript functionality."""
    return """
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;

            // Hide all tab content
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].classList.remove("active");
            }

            // Remove active class from all tab buttons
            tablinks = document.getElementsByClassName("tab-button");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].classList.remove("active");
            }

            // Show the selected tab and mark button as active
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
        }

        // Initialize first tab as active on page load
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelector('.tab-button').click();
        });
    """