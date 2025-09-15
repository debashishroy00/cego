"""Results aggregation and statistical analysis for CEGO benchmark."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import json

from ..runners.metrics import BenchmarkMetrics, calculate_expected_calibration_error
from ..runners.adapters import OptimizerType


def aggregate_metrics(metrics_list: List[BenchmarkMetrics]) -> Dict[str, Any]:
    """Aggregate metrics across multiple benchmark runs.

    Args:
        metrics_list: List of benchmark metrics

    Returns:
        Dictionary with aggregated statistics
    """
    if not metrics_list:
        return {}

    # Separate by optimizer type
    by_optimizer = defaultdict(list)
    for metric in metrics_list:
        by_optimizer[metric.optimizer].append(metric)

    aggregated = {}

    for optimizer, opt_metrics in by_optimizer.items():
        if not opt_metrics:
            continue

        # Filter successful runs for most metrics
        successful = [m for m in opt_metrics if m.success]

        stats = {
            "total_runs": len(opt_metrics),
            "successful_runs": len(successful),
            "success_rate": len(successful) / len(opt_metrics) if opt_metrics else 0.0,
        }

        if successful:
            # Token reduction metrics
            reductions = [m.token_reduction_percentage for m in successful
                         if m.token_reduction_percentage is not None]
            if reductions:
                stats.update({
                    "reduction_mean": np.mean(reductions),
                    "reduction_median": np.median(reductions),
                    "reduction_std": np.std(reductions),
                    "reduction_min": np.min(reductions),
                    "reduction_max": np.max(reductions),
                    "reduction_p90": np.percentile(reductions, 90),
                })

            # Semantic retention metrics
            retentions = [m.semantic_retention for m in successful
                         if m.semantic_retention is not None]
            if retentions:
                stats.update({
                    "retention_mean": np.mean(retentions),
                    "retention_median": np.median(retentions),
                    "retention_std": np.std(retentions),
                    "retention_min": np.min(retentions),
                    "retention_max": np.max(retentions),
                    "retention_p10": np.percentile(retentions, 10),
                })

            # Latency metrics
            latencies = [m.latency_ms for m in successful]
            if latencies:
                stats.update({
                    "latency_mean": np.mean(latencies),
                    "latency_median": np.median(latencies),
                    "latency_std": np.std(latencies),
                    "latency_min": np.min(latencies),
                    "latency_max": np.max(latencies),
                    "latency_p90": np.percentile(latencies, 90),
                    "latency_p95": np.percentile(latencies, 95),
                })

            # Confidence metrics
            confidences = [m.confidence for m in successful
                          if m.confidence is not None]
            if confidences:
                stats.update({
                    "confidence_mean": np.mean(confidences),
                    "confidence_median": np.median(confidences),
                    "confidence_std": np.std(confidences),
                })

            # Junk handling metrics
            junk_rates = [m.junk_kept_rate for m in successful
                         if m.junk_kept_rate is not None]
            if junk_rates:
                stats.update({
                    "junk_kept_rate_mean": np.mean(junk_rates),
                    "junk_kept_rate_median": np.median(junk_rates),
                    "junk_kept_rate_max": np.max(junk_rates),
                })

            # Intent preservation
            intent_scores = [m.intent_preservation for m in successful
                           if m.intent_preservation is not None]
            if intent_scores:
                stats.update({
                    "intent_preservation_mean": np.mean(intent_scores),
                    "intent_preservation_rate": sum(1 for s in intent_scores if s >= 0.95) / len(intent_scores),
                })

            # Precision/Recall/F1 (where available)
            precisions = [m.precision for m in successful if m.precision is not None]
            recalls = [m.recall for m in successful if m.recall is not None]
            f1s = [m.f1_score for m in successful if m.f1_score is not None]

            if precisions:
                stats["precision_mean"] = np.mean(precisions)
            if recalls:
                stats["recall_mean"] = np.mean(recalls)
            if f1s:
                stats["f1_mean"] = np.mean(f1s)

        # Error analysis
        failed = [m for m in opt_metrics if not m.success]
        if failed:
            error_counts = defaultdict(int)
            for metric in failed:
                if metric.error_message:
                    # Categorize common errors
                    error_msg = metric.error_message.lower()
                    if "timeout" in error_msg:
                        error_counts["timeout"] += 1
                    elif "connection" in error_msg:
                        error_counts["connection"] += 1
                    elif "http" in error_msg:
                        error_counts["http_error"] += 1
                    else:
                        error_counts["other"] += 1
                else:
                    error_counts["unknown"] += 1

            stats["error_breakdown"] = dict(error_counts)

        aggregated[optimizer.value] = stats

    # Calculate calibration error
    all_successful = [m for m in metrics_list if m.success]
    if all_successful:
        aggregated["calibration"] = {
            "expected_calibration_error": calculate_expected_calibration_error(all_successful)
        }

    return aggregated


def aggregate_by_domain(metrics_list: List[BenchmarkMetrics],
                       domain_mapping: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Aggregate metrics by domain.

    Args:
        metrics_list: List of benchmark metrics
        domain_mapping: Mapping from test_id to domain name

    Returns:
        Dictionary mapping domain to aggregated stats
    """
    # Group metrics by domain
    by_domain = defaultdict(list)
    for metric in metrics_list:
        domain = domain_mapping.get(metric.test_id, "unknown")
        by_domain[domain].append(metric)

    # Aggregate each domain
    domain_stats = {}
    for domain, domain_metrics in by_domain.items():
        domain_stats[domain] = aggregate_metrics(domain_metrics)

    return domain_stats


def create_summary_table(aggregated_stats: Dict[str, Any]) -> pd.DataFrame:
    """Create summary table from aggregated statistics.

    Args:
        aggregated_stats: Aggregated statistics dictionary

    Returns:
        Pandas DataFrame with summary statistics
    """
    rows = []

    for optimizer, stats in aggregated_stats.items():
        if optimizer == "calibration":
            continue

        row = {
            "optimizer": optimizer,
            "success_rate": stats.get("success_rate", 0.0),
            "reduction_median": stats.get("reduction_median"),
            "reduction_mean": stats.get("reduction_mean"),
            "retention_mean": stats.get("retention_mean"),
            "retention_min": stats.get("retention_min"),
            "latency_p90": stats.get("latency_p90"),
            "junk_kept_rate_median": stats.get("junk_kept_rate_median"),
            "intent_preservation_rate": stats.get("intent_preservation_rate"),
            "confidence_mean": stats.get("confidence_mean"),
            "total_runs": stats.get("total_runs", 0),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def create_calibration_table(metrics_list: List[BenchmarkMetrics],
                           num_bins: int = 10) -> pd.DataFrame:
    """Create calibration analysis table.

    Args:
        metrics_list: List of benchmark metrics
        num_bins: Number of confidence bins

    Returns:
        DataFrame with calibration statistics per bin
    """
    # Filter successful results with confidence and retention
    valid_metrics = [
        m for m in metrics_list
        if m.success and m.confidence is not None and m.semantic_retention is not None
    ]

    if not valid_metrics:
        return pd.DataFrame()

    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    rows = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find metrics in this bin
        in_bin = [
            m for m in valid_metrics
            if bin_lower <= m.confidence < bin_upper or
               (bin_upper == 1.0 and m.confidence == 1.0)
        ]

        if not in_bin:
            continue

        avg_confidence = np.mean([m.confidence for m in in_bin])
        avg_accuracy = np.mean([m.semantic_retention for m in in_bin])
        count = len(in_bin)

        rows.append({
            "bin_lower": bin_lower,
            "bin_upper": bin_upper,
            "count": count,
            "avg_confidence": avg_confidence,
            "avg_accuracy": avg_accuracy,
            "calibration_error": abs(avg_confidence - avg_accuracy)
        })

    return pd.DataFrame(rows)


def save_results(metrics_list: List[BenchmarkMetrics],
                output_dir: Path,
                run_id: str,
                domain_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Path]:
    """Save benchmark results to files.

    Args:
        metrics_list: List of benchmark metrics
        output_dir: Output directory
        run_id: Run identifier
        domain_mapping: Optional mapping from test_id to domain

    Returns:
        Dictionary mapping file type to saved file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Save detailed results as JSONL
    results_file = output_dir / f"{run_id}_results.jsonl"
    with open(results_file, 'w', encoding='utf-8') as f:
        for metric in metrics_list:
            # Convert to dictionary for JSON serialization
            result_dict = {
                "test_id": metric.test_id,
                "optimizer": metric.optimizer.value,
                "success": metric.success,
                "input_tokens": metric.input_tokens,
                "output_tokens": metric.output_tokens,
                "token_reduction_percentage": metric.token_reduction_percentage,
                "semantic_retention": metric.semantic_retention,
                "confidence": metric.confidence,
                "latency_ms": metric.latency_ms,
                "kept_indices": metric.kept_indices,
                "junk_kept_rate": metric.junk_kept_rate,
                "intent_preservation": metric.intent_preservation,
                "precision": metric.precision,
                "recall": metric.recall,
                "f1_score": metric.f1_score,
                "error_message": metric.error_message,
            }

            if domain_mapping:
                result_dict["domain"] = domain_mapping.get(metric.test_id, "unknown")

            f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')

    saved_files["results"] = results_file

    # Aggregate and save summary
    aggregated_stats = aggregate_metrics(metrics_list)
    summary_table = create_summary_table(aggregated_stats)

    if not summary_table.empty:
        summary_file = output_dir / f"{run_id}_summary.csv"
        summary_table.to_csv(summary_file, index=False)
        saved_files["summary"] = summary_file

    # Save calibration analysis
    calibration_table = create_calibration_table(metrics_list)
    if not calibration_table.empty:
        calibration_file = output_dir / f"{run_id}_calibration.csv"
        calibration_table.to_csv(calibration_file, index=False)
        saved_files["calibration"] = calibration_file

    # Save domain breakdown if available
    if domain_mapping:
        domain_stats = aggregate_by_domain(metrics_list, domain_mapping)
        domain_file = output_dir / f"{run_id}_by_domain.json"
        with open(domain_file, 'w', encoding='utf-8') as f:
            json.dump(domain_stats, f, indent=2, ensure_ascii=False)
        saved_files["domains"] = domain_file

    # Save aggregated statistics
    stats_file = output_dir / f"{run_id}_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated_stats, f, indent=2, ensure_ascii=False, default=str)
    saved_files["stats"] = stats_file

    return saved_files