"""Main CLI orchestrator for CEGO benchmark harness."""

import argparse
import json
import logging
import random
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from .loaders import load_jsonl_dataset, validate_dataset, filter_test_cases, TestCase
from .adapters import CEGOAdapter, create_adapter_from_config, OptimizerType
from .metrics import calculate_benchmark_metrics, BenchmarkMetrics
from ..reporting.aggregate import save_results
from ..reporting.html_report import generate_html_report
from ..scorers.embed_scorer import is_available as embed_available
from ..utils.timing import timer
from ..utils.hashing import hash_dict


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AcceptanceGates:
    """Acceptance gate evaluation for benchmark results."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize acceptance gates.

        Args:
            config: Configuration dictionary with gate thresholds
        """
        self.config = config

    def evaluate(self, aggregated_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all acceptance gates.

        Args:
            aggregated_stats: Aggregated benchmark statistics

        Returns:
            Dictionary with gate results and overall pass/fail
        """
        results = {}
        all_passed = True

        # Entropy optimizer gates
        entropy_stats = aggregated_stats.get("entropy", {})
        if entropy_stats:
            # Median reduction >= 60%
            entropy_reduction = entropy_stats.get("reduction_median", 0) * 100  # Convert to percentage
            entropy_reduction_gate = {
                "passed": entropy_reduction >= 60.0,
                "actual": f"{entropy_reduction:.1f}%",
                "required": ">=60%"
            }
            results["entropy_reduction"] = entropy_reduction_gate
            if not entropy_reduction_gate["passed"]:
                all_passed = False

            # Mean retention >= 0.92
            entropy_retention = entropy_stats.get("retention_mean", 0)
            entropy_retention_gate = {
                "passed": entropy_retention >= 0.92,
                "actual": f"{entropy_retention:.3f}",
                "required": ">=0.92"
            }
            results["entropy_retention"] = entropy_retention_gate
            if not entropy_retention_gate["passed"]:
                all_passed = False

            # Junk kept rate <= 10%
            entropy_junk_rate = entropy_stats.get("junk_kept_rate_median", 0)
            if entropy_junk_rate is not None:
                entropy_junk_gate = {
                    "passed": entropy_junk_rate <= 0.10,
                    "actual": f"{entropy_junk_rate*100:.1f}%",
                    "required": "<=10%"
                }
                results["entropy_junk_rate"] = entropy_junk_gate
                if not entropy_junk_gate["passed"]:
                    all_passed = False

        # Pattern optimizer gates
        pattern_stats = aggregated_stats.get("pattern", {})
        if pattern_stats:
            # Median reduction >= 20%
            pattern_reduction = pattern_stats.get("reduction_median", 0) * 100  # Convert to percentage
            pattern_reduction_gate = {
                "passed": pattern_reduction >= 20.0,
                "actual": f"{pattern_reduction:.1f}%",
                "required": ">=20%"
            }
            results["pattern_reduction"] = pattern_reduction_gate
            if not pattern_reduction_gate["passed"]:
                all_passed = False

            # Mean retention >= 0.90
            pattern_retention = pattern_stats.get("retention_mean", 0)
            pattern_retention_gate = {
                "passed": pattern_retention >= 0.90,
                "actual": f"{pattern_retention:.3f}",
                "required": ">=0.90"
            }
            results["pattern_retention"] = pattern_retention_gate
            if not pattern_retention_gate["passed"]:
                all_passed = False

            # Junk kept rate <= 15%
            pattern_junk_rate = pattern_stats.get("junk_kept_rate_median", 0)
            if pattern_junk_rate is not None:
                pattern_junk_gate = {
                    "passed": pattern_junk_rate <= 0.15,
                    "actual": f"{pattern_junk_rate*100:.1f}%",
                    "required": "<=15%"
                }
                results["pattern_junk_rate"] = pattern_junk_gate
                if not pattern_junk_gate["passed"]:
                    all_passed = False

        # Intent preservation (both optimizers)
        for optimizer in ["entropy", "pattern"]:
            opt_stats = aggregated_stats.get(optimizer, {})
            intent_rate = opt_stats.get("intent_preservation_rate", 0)
            if intent_rate is not None:
                intent_gate = {
                    "passed": intent_rate >= 0.95,
                    "actual": f"{intent_rate*100:.1f}%",
                    "required": ">=95%"
                }
                results[f"{optimizer}_intent_preservation"] = intent_gate
                if not intent_gate["passed"]:
                    all_passed = False

        # Latency comparison
        if entropy_stats and pattern_stats:
            entropy_latency = entropy_stats.get("latency_p90", float('inf'))
            pattern_latency = pattern_stats.get("latency_p90", float('inf'))

            if entropy_latency < float('inf') and pattern_latency < float('inf'):
                latency_ratio = entropy_latency / pattern_latency
                latency_gate = {
                    "passed": latency_ratio <= 1.25,
                    "actual": f"{latency_ratio:.2f}x",
                    "required": "<=1.25x"
                }
                results["latency_ratio"] = latency_gate
                if not latency_gate["passed"]:
                    all_passed = False

        results["overall_passed"] = all_passed
        return results


class SuggestedTweaks:
    """Generate suggested parameter tweaks when gates fail."""

    def generate(self, aggregated_stats: Dict[str, Any], gate_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate suggested tweaks based on failed gates.

        Args:
            aggregated_stats: Aggregated benchmark statistics
            gate_results: Acceptance gate results

        Returns:
            Dictionary with suggested parameter adjustments
        """
        tweaks = {}

        entropy_stats = aggregated_stats.get("entropy", {})
        pattern_stats = aggregated_stats.get("pattern", {})

        # Entropy optimizer tweaks
        if not gate_results.get("entropy_reduction", {}).get("passed", True):
            # Increase aggressiveness
            tweaks["lambda_decay"] = "0.88  # was 0.92, more aggressive pruning"
            tweaks["phase_min_gain"] = "0.010  # was 0.015, lower threshold for phase detection"

        if not gate_results.get("entropy_retention", {}).get("passed", True):
            # Reduce aggressiveness
            tweaks["lambda_decay"] = "0.95  # was 0.92, less aggressive pruning"
            tweaks["mmr_weight"] = "0.6  # was 0.5, more diversity"

        if not gate_results.get("entropy_junk_rate", {}).get("passed", True):
            # Be more aggressive with junk
            tweaks["junk_soft_threshold"] = "0.42  # was 0.35, higher junk threshold"
            tweaks["hard_drop_threshold"] = "0.15  # was 0.20, more aggressive hard drops"

        # Pattern optimizer tweaks
        if not gate_results.get("pattern_reduction", {}).get("passed", True):
            tweaks["pattern_keep_ratio"] = "0.3  # was 0.4, keep fewer items"

        if not gate_results.get("pattern_retention", {}).get("passed", True):
            tweaks["pattern_keep_ratio"] = "0.5  # was 0.4, keep more items"

        if not gate_results.get("pattern_junk_rate", {}).get("passed", True):
            tweaks["pattern_junk_threshold"] = "0.45  # was 0.40, higher junk threshold"

        # Intent preservation tweaks
        intent_issues = any(
            not gate_results.get(f"{opt}_intent_preservation", {}).get("passed", True)
            for opt in ["entropy", "pattern"]
        )
        if intent_issues:
            tweaks["domain_boost_factor"] = "1.3  # was 1.0, boost primary domain"
            tweaks["soft_keep_threshold"] = "0.25  # was 0.30, softer intent preservation"

        # Latency tweaks
        if not gate_results.get("latency_ratio", {}).get("passed", True):
            tweaks["entropy_timeout"] = "15.0  # was 30.0, shorter timeout for entropy"
            tweaks["tfidf_fallback_threshold"] = "0.5  # was 0.3, earlier TF-IDF fallback"

        return tweaks


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Set defaults
    config.setdefault('endpoints', {
        'pattern': 'http://127.0.0.1:8003/optimize/pattern',
        'entropy': 'http://127.0.0.1:8003/optimize/entropy'
    })

    config.setdefault('datasets', [
        'datasets/insurance.jsonl',
        'datasets/sdlc.jsonl',
        'datasets/mixed.jsonl',
        'datasets/noisy.jsonl',
        'datasets/dupes.jsonl',
        'datasets/format.jsonl'
    ])

    config.setdefault('scoring', {
        'method_order': ['embed', 'tfidf', 'rouge'],
        'embed_model': 'all-MiniLM-L6-v2'
    })

    config.setdefault('runs', {
        'repeat': 3,
        'max_items': 50
    })

    config.setdefault('reporting', {
        'save_html': True,
        'save_artifacts': True
    })

    return config


def run_benchmark(config: Dict[str, Any],
                 dataset_paths: Optional[List[Path]] = None,
                 output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run the complete benchmark harness.

    Args:
        config: Benchmark configuration
        dataset_paths: Optional override of dataset paths
        output_dir: Optional override of output directory

    Returns:
        Dictionary with run results and file paths
    """
    # Set random seed for reproducibility
    random.seed(42)

    # Setup output directory
    if output_dir is None:
        output_dir = Path("output/runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    dataset_paths = dataset_paths or [Path(p) for p in config['datasets']]
    all_test_cases = []
    domain_mapping = {}

    logger.info(f"Loading {len(dataset_paths)} datasets...")

    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            continue

        try:
            test_cases = load_jsonl_dataset(dataset_path)
            domain_name = dataset_path.stem  # Use filename as domain

            # Validate dataset
            stats = validate_dataset(test_cases)
            logger.info(f"Loaded {dataset_path.name}: {stats['total_cases']} cases, {stats['total_items']} items")

            if stats['warnings']:
                logger.warning(f"Dataset warnings for {dataset_path.name}: {', '.join(stats['warnings'])}")

            # Filter test cases
            filtered_cases = filter_test_cases(
                test_cases,
                max_items=config['runs'].get('max_items'),
                require_gold=False,
                require_junk_labels=False
            )

            logger.info(f"Using {len(filtered_cases)} test cases from {dataset_path.name}")

            # Add to collection
            all_test_cases.extend(filtered_cases)

            # Track domain mapping
            for test_case in filtered_cases:
                domain_mapping[test_case.id] = domain_name

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_path}: {e}")
            continue

    if not all_test_cases:
        raise ValueError("No valid test cases found in any dataset")

    logger.info(f"Total test cases: {len(all_test_cases)}")

    # Initialize adapter
    adapter = create_adapter_from_config(config)

    # Health check
    logger.info("Performing health check...")
    health = adapter.health_check()
    for optimizer, is_healthy in health.items():
        status = "OK" if is_healthy else "FAILED"
        logger.info(f"  {optimizer.value}: {status}")

    if not any(health.values()):
        raise RuntimeError("No optimizers are healthy - check endpoint URLs and server status")

    # Run benchmarks
    all_metrics = []
    total_runs = len(all_test_cases) * len([OptimizerType.PATTERN, OptimizerType.ENTROPY]) * config['runs']['repeat']

    logger.info(f"Starting benchmark with {total_runs} total runs...")

    with tqdm(total=total_runs, desc="Running benchmarks") as pbar:
        for test_case in all_test_cases:
            for optimizer in [OptimizerType.PATTERN, OptimizerType.ENTROPY]:
                for run_idx in range(config['runs']['repeat']):
                    try:
                        with timer() as t:
                            result = adapter.optimize(test_case, optimizer)

                        # Calculate metrics
                        metrics = calculate_benchmark_metrics(test_case, result)

                        all_metrics.append(metrics)

                        # Log progress
                        if result.success:
                            reduction = result.token_reduction_percentage or 0
                            retention = result.semantic_retention or 0
                            logger.debug(
                                f"{test_case.id} ({optimizer.value}): "
                                f"reduction={reduction:.1f}%, retention={retention:.3f}, "
                                f"latency={result.latency_ms:.1f}ms"
                            )
                        else:
                            logger.warning(
                                f"{test_case.id} ({optimizer.value}) FAILED: {result.error_message}"
                            )

                    except Exception as e:
                        logger.error(f"Benchmark failed for {test_case.id} ({optimizer.value}): {e}")

                    pbar.update(1)

    adapter.close()

    logger.info(f"Benchmark complete. {len(all_metrics)} results collected.")

    # Generate reports
    logger.info("Generating reports...")

    # Create run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_hash = hash_dict(config)[:8]
    run_id = f"{run_id}_{config_hash}"

    # Save detailed results
    saved_files = save_results(all_metrics, output_dir, run_id, domain_mapping)

    # Evaluate acceptance gates
    from ..reporting.aggregate import aggregate_metrics
    aggregated_stats = aggregate_metrics(all_metrics)

    gates = AcceptanceGates(config)
    gate_results = gates.evaluate(aggregated_stats)

    # Generate tweaks if needed
    suggested_tweaks = None
    if not gate_results.get("overall_passed", False):
        tweaks_gen = SuggestedTweaks()
        suggested_tweaks = tweaks_gen.generate(aggregated_stats, gate_results)

        # Save suggested tweaks
        if suggested_tweaks:
            tweaks_file = output_dir / f"{run_id}_suggested_tweaks.yaml"
            with open(tweaks_file, 'w', encoding='utf-8') as f:
                f.write("# Suggested parameter tweaks to meet acceptance gates\\n\\n")
                for key, value in suggested_tweaks.items():
                    f.write(f"{key}: {value}\\n")
            saved_files["tweaks"] = tweaks_file

    # Generate HTML report
    if config['reporting']['save_html']:
        html_file = output_dir / f"{run_id}_report.html"
        generate_html_report(
            all_metrics,
            html_file,
            config,
            domain_mapping,
            gate_results,
            suggested_tweaks
        )
        saved_files["html_report"] = html_file

    # Print summary
    logger.info("\\n" + "="*60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*60)

    successful_metrics = [m for m in all_metrics if m.success]
    success_rate = len(successful_metrics) / len(all_metrics) * 100

    logger.info(f"Total runs: {len(all_metrics)}")
    logger.info(f"Success rate: {success_rate:.1f}%")

    # Optimizer-specific summary
    for optimizer in [OptimizerType.PATTERN, OptimizerType.ENTROPY]:
        opt_metrics = [m for m in successful_metrics if m.optimizer == optimizer]
        if opt_metrics:
            reductions = [m.token_reduction_percentage for m in opt_metrics if m.token_reduction_percentage]
            retentions = [m.semantic_retention for m in opt_metrics if m.semantic_retention]
            latencies = [m.latency_ms for m in opt_metrics]

            if reductions and retentions:
                logger.info(f"\\n{optimizer.value.upper()}:")
                median_reduction = sorted(reductions)[len(reductions)//2] * 100  # Convert to percentage
                logger.info(f"  Median reduction: {median_reduction:.1f}%")
                logger.info(f"  Mean retention: {sum(retentions)/len(retentions):.3f}")
                logger.info(f"  P90 latency: {sorted(latencies)[int(len(latencies)*0.9)]:.1f}ms")

    # Acceptance gates summary
    logger.info(f"\\nACCEPTANCE GATES: {'PASS' if gate_results.get('overall_passed') else 'FAIL'}")
    failed_gates = [name for name, result in gate_results.items()
                   if isinstance(result, dict) and not result.get('passed', True)]
    if failed_gates:
        logger.info(f"Failed gates: {', '.join(failed_gates)}")

    logger.info(f"\\nReports saved to: {output_dir}")
    for report_type, file_path in saved_files.items():
        logger.info(f"  {report_type}: {file_path.name}")

    logger.info("="*60)

    return {
        "run_id": run_id,
        "output_dir": output_dir,
        "saved_files": saved_files,
        "metrics": all_metrics,
        "aggregated_stats": aggregated_stats,
        "gate_results": gate_results,
        "suggested_tweaks": suggested_tweaks
    }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CEGO Benchmark Harness - Evaluate optimization performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cego_bench.runners.run_bench --config configs/default.yaml
  python -m cego_bench.runners.run_bench --config configs/stress.yaml --no-embed
  python -m cego_bench.runners.run_bench --dataset datasets/mixed.jsonl --repeat 1
        """
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to configuration file (default: configs/default.yaml)"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        help="Override dataset path(s). Can be specified multiple times."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: auto-generated in output/runs/)"
    )

    parser.add_argument(
        "--repeat",
        type=int,
        help="Override number of repetitions per test"
    )

    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Disable embedding-based scoring (fall back to TF-IDF)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Apply CLI overrides
        if args.repeat is not None:
            config['runs']['repeat'] = args.repeat

        if args.no_embed:
            config['scoring']['method_order'] = ['tfidf', 'rouge']

        # Check embedding availability
        if not args.no_embed and 'embed' in config['scoring']['method_order']:
            if not embed_available():
                logger.warning("Embedding scorer not available, falling back to TF-IDF")
                config['scoring']['method_order'] = [
                    m for m in config['scoring']['method_order'] if m != 'embed'
                ]

        # Run benchmark
        results = run_benchmark(
            config,
            dataset_paths=args.dataset,
            output_dir=args.output_dir
        )

        # Exit with appropriate code
        if results['gate_results'].get('overall_passed', False):
            logger.info("All acceptance gates passed!")
            sys.exit(0)
        else:
            logger.warning("Some acceptance gates failed.")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()