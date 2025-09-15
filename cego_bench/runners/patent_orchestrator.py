"""Patent evidence report orchestrator - complete pipeline for CEGO patent pack generation."""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from .loaders import load_jsonl_dataset, filter_test_cases
from .adapters import create_adapter_from_config
from .run_bench import run_benchmark, AcceptanceGates, SuggestedTweaks
from .ablations import AblationFramework
from ..reporting.patent_report import generate_patent_report
from ..reporting.exemplars import ExemplarGenerator
from ..reporting.aggregate import save_results


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatentOrchestrator:
    """Orchestrates complete patent evidence generation pipeline."""

    def __init__(self, output_base_dir: Path = None):
        """Initialize patent orchestrator.

        Args:
            output_base_dir: Base directory for outputs (default: output/patent_runs)
        """
        if output_base_dir is None:
            output_base_dir = Path("output/patent_runs")

        self.output_base_dir = Path(output_base_dir)
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_base_dir / f"patent_run_{self.run_timestamp}"

        # Create directory structure
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.patent_pack_dir = self.run_dir / "patent_pack"
        self.exemplars_dir = self.run_dir / "exemplars"
        self.configs_snapshot_dir = self.run_dir / "configs_snapshot"
        self.ablations_dir = self.run_dir / "ablations"

        for dir_path in [self.patent_pack_dir, self.exemplars_dir,
                        self.configs_snapshot_dir, self.ablations_dir]:
            dir_path.mkdir(exist_ok=True)

        logger.info(f"Patent orchestrator initialized: {self.run_dir}")

    def generate_full_patent_evidence(self,
                                    config_path: Path,
                                    run_ablations: bool = True,
                                    run_stress_test: bool = True,
                                    generate_pdfs: bool = False) -> Dict[str, Path]:
        """Generate complete patent evidence pack.

        Args:
            config_path: Path to benchmark configuration
            run_ablations: Whether to run ablation studies
            run_stress_test: Whether to run stress tests
            generate_pdfs: Whether to generate PDF versions of annexes

        Returns:
            Dictionary mapping deliverable types to file paths
        """
        logger.info("Starting complete patent evidence generation...")

        # Step 1: Load configuration and create snapshot
        config = self._load_and_snapshot_config(config_path)

        # Step 2: Run primary benchmark sweep
        logger.info("Running primary benchmark sweep...")
        primary_results = self._run_primary_benchmark(config)

        # Step 3: Run stress test if requested
        stress_results = None
        if run_stress_test:
            logger.info("Running stress test benchmark...")
            stress_results = self._run_stress_benchmark(config)

        # Step 4: Run ablation studies if requested
        ablation_results = None
        if run_ablations:
            logger.info("Running ablation studies...")
            ablation_results = self._run_ablation_studies(config, primary_results['metrics'])

        # Step 5: Generate exemplars
        logger.info("Generating representative exemplars...")
        exemplar_files = self._generate_exemplars(primary_results, config)

        # Step 6: Generate patent report pack
        logger.info("Generating patent report pack...")
        report_files = self._generate_patent_reports(
            primary_results, stress_results, ablation_results, config
        )

        # Step 7: Generate PDFs if requested
        if generate_pdfs:
            logger.info("Converting annexes to PDF...")
            pdf_files = self._generate_pdfs(report_files)
        else:
            pdf_files = {}

        # Step 8: Create final deliverable package
        deliverables = self._package_deliverables(
            report_files, exemplar_files, pdf_files, primary_results
        )

        logger.info(f"Patent evidence generation complete!")
        logger.info(f"Deliverables saved to: {self.run_dir}")

        return deliverables

    def _load_and_snapshot_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration and create reproducibility snapshot.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        import yaml

        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Create snapshot
        snapshot = {
            "config": config,
            "config_file": str(config_path),
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": str(Path.cwd()),
        }

        # Try to get git info
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                              stderr=subprocess.DEVNULL).decode().strip()
            git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                                stderr=subprocess.DEVNULL).decode().strip()
            snapshot["git"] = {
                "commit": git_hash,
                "branch": git_branch
            }
        except:
            snapshot["git"] = "unavailable"

        # Try to get package versions
        try:
            import pkg_resources
            packages = {}
            for package_name in ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'requests']:
                try:
                    packages[package_name] = pkg_resources.get_distribution(package_name).version
                except:
                    packages[package_name] = "not installed"
            snapshot["packages"] = packages
        except:
            snapshot["packages"] = "unavailable"

        # Save snapshot
        snapshot_file = self.configs_snapshot_dir / "environment_snapshot.json"
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, default=str)

        # Copy original config
        shutil.copy2(config_path, self.configs_snapshot_dir / config_path.name)

        return config

    def _run_primary_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run primary benchmark sweep.

        Args:
            config: Benchmark configuration

        Returns:
            Dictionary with metrics, aggregates, and gate results
        """
        # Override output directory
        config['output'] = {'base_dir': str(self.run_dir), 'include_timestamp': False}

        # Run benchmark
        from .run_bench import run_benchmark
        results = run_benchmark(
            config,
            output_dir=self.run_dir / "primary_benchmark"
        )

        return {
            'metrics': results['metrics'],
            'aggregated_stats': results['aggregated_stats'],
            'gate_results': results['gate_results'],
            'suggested_tweaks': results.get('suggested_tweaks'),
            'saved_files': results['saved_files']
        }

    def _run_stress_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run stress test benchmark.

        Args:
            config: Base configuration

        Returns:
            Stress test results
        """
        # Create stress config
        stress_config = config.copy()
        stress_config['runs']['repeat'] = 1  # Faster for stress test
        stress_config['runs']['max_items'] = 100  # Allow larger inputs
        stress_config['timeout'] = 60  # Longer timeout

        # Run stress benchmark
        results = run_benchmark(
            stress_config,
            output_dir=self.run_dir / "stress_benchmark"
        )

        return {
            'metrics': results['metrics'],
            'aggregated_stats': results['aggregated_stats'],
            'gate_results': results['gate_results']
        }

    def _run_ablation_studies(self, config: Dict[str, Any], baseline_metrics: List) -> Dict[str, Any]:
        """Run ablation studies.

        Args:
            config: Base configuration
            baseline_metrics: Baseline metrics for comparison

        Returns:
            Ablation results
        """
        # Load test cases for ablation (use subset)
        all_test_cases = []
        for dataset_path in config.get('datasets', []):
            try:
                test_cases = load_jsonl_dataset(Path(dataset_path))
                filtered_cases = filter_test_cases(test_cases, max_items=20)
                all_test_cases.extend(filtered_cases[:2])  # Limit for ablations
            except Exception as e:
                logger.warning(f"Could not load dataset {dataset_path}: {e}")

        if not all_test_cases:
            logger.warning("No test cases available for ablation studies")
            return {}

        # Run ablations
        framework = AblationFramework(config, self.ablations_dir)

        # Select key ablations for patent evidence
        selected_ablations = [
            "lambda_decay_0.88", "lambda_decay_0.95",
            "mmr_weight_0.3", "mmr_weight_0.7",
            "junk_soft_threshold_0.30", "junk_soft_threshold_0.40"
        ]

        ablation_results = framework.run_ablation_study(
            all_test_cases,
            baseline_results=baseline_metrics,
            selected_ablations=selected_ablations
        )

        return ablation_results

    def _generate_exemplars(self, primary_results: Dict, config: Dict) -> Dict[str, List[Path]]:
        """Generate representative exemplars.

        Args:
            primary_results: Primary benchmark results
            config: Configuration

        Returns:
            Dictionary mapping domains to exemplar file paths
        """
        # Load all test cases
        all_test_cases = []
        domain_mapping = {}

        for dataset_path in config.get('datasets', []):
            try:
                test_cases = load_jsonl_dataset(Path(dataset_path))
                domain_name = Path(dataset_path).stem
                all_test_cases.extend(test_cases)

                # Build domain mapping
                for tc in test_cases:
                    domain_mapping[tc.id] = domain_name

            except Exception as e:
                logger.warning(f"Could not load dataset {dataset_path}: {e}")

        # Generate exemplars
        generator = ExemplarGenerator(self.exemplars_dir)
        exemplar_files = generator.generate_exemplars(
            primary_results['metrics'],
            all_test_cases,
            domain_mapping,
            top_k=3
        )

        # Create summary
        generator.create_domain_summary(exemplar_files)

        return exemplar_files

    def _generate_patent_reports(self,
                               primary_results: Dict,
                               stress_results: Optional[Dict],
                               ablation_results: Optional[Dict],
                               config: Dict) -> Dict[str, Path]:
        """Generate patent report pack.

        Args:
            primary_results: Primary benchmark results
            stress_results: Optional stress test results
            ablation_results: Optional ablation results
            config: Configuration

        Returns:
            Dictionary mapping report types to file paths
        """
        # Build domain mapping
        domain_mapping = {}
        for dataset_path in config.get('datasets', []):
            try:
                test_cases = load_jsonl_dataset(Path(dataset_path))
                domain_name = Path(dataset_path).stem
                for tc in test_cases:
                    domain_mapping[tc.id] = domain_name
            except:
                continue

        # Generate main patent report
        report_files = generate_patent_report(
            self.patent_pack_dir,
            f"patent_{self.run_timestamp}",
            primary_results['metrics'],
            config,
            domain_mapping,
            primary_results['gate_results'],
            ablation_results
        )

        return report_files

    def _generate_pdfs(self, report_files: Dict[str, Path]) -> Dict[str, Path]:
        """Generate PDF versions of markdown annexes.

        Args:
            report_files: Dictionary of report files

        Returns:
            Dictionary of generated PDF files
        """
        pdf_files = {}

        try:
            # Try to use pandoc if available
            subprocess.run(['pandoc', '--version'], check=True, capture_output=True)
            pandoc_available = True
        except:
            pandoc_available = False
            logger.warning("Pandoc not available, skipping PDF generation")

        if pandoc_available:
            for report_name, report_path in report_files.items():
                if report_path.suffix == '.md':
                    pdf_path = report_path.with_suffix('.pdf')
                    try:
                        subprocess.run([
                            'pandoc', str(report_path),
                            '-o', str(pdf_path),
                            '--pdf-engine=pdflatex',
                            '--variable', 'geometry:margin=1in'
                        ], check=True, capture_output=True)
                        pdf_files[report_name] = pdf_path
                        logger.info(f"Generated PDF: {pdf_path.name}")
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Failed to generate PDF for {report_name}: {e}")

        return pdf_files

    def _package_deliverables(self,
                            report_files: Dict[str, Path],
                            exemplar_files: Dict[str, List[Path]],
                            pdf_files: Dict[str, Path],
                            primary_results: Dict) -> Dict[str, Path]:
        """Package final deliverables.

        Args:
            report_files: Report files
            exemplar_files: Exemplar files
            pdf_files: PDF files
            primary_results: Primary benchmark results

        Returns:
            Dictionary of deliverable paths
        """
        deliverables = {}

        # Main patent pack
        deliverables.update(report_files)
        deliverables.update(pdf_files)

        # Exemplars
        deliverables['exemplars_dir'] = self.exemplars_dir

        # Raw data
        deliverables['configs_snapshot'] = self.configs_snapshot_dir
        deliverables['ablations_dir'] = self.ablations_dir

        # Create manifest
        manifest = {
            "run_id": f"patent_{self.run_timestamp}",
            "generated": datetime.now().isoformat(),
            "deliverables": {
                "patent_pack": str(self.patent_pack_dir),
                "exemplars": str(self.exemplars_dir),
                "configs_snapshot": str(self.configs_snapshot_dir),
                "ablations": str(self.ablations_dir),
                "raw_results": str(primary_results.get('saved_files', {}))
            },
            "files": {
                name: str(path) for name, path in deliverables.items()
            },
            "statistics": {
                "total_tests": len(primary_results['metrics']),
                "successful_tests": len([m for m in primary_results['metrics'] if m.success]),
                "domains_tested": len(set(exemplar_files.keys())),
                "exemplars_generated": sum(len(files) for files in exemplar_files.values())
            }
        }

        manifest_file = self.run_dir / "PATENT_MANIFEST.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, default=str)

        deliverables['manifest'] = manifest_file

        # Create README
        readme_content = f"""# CEGO Patent Evidence Pack

**Run ID:** patent_{self.run_timestamp}
**Generated:** {datetime.now().isoformat()}

## Contents

### Patent Pack (patent_pack/)
- AnnexA_Executive_Summary.md - Key findings and gate results
- AnnexB_Methodology.md - Experimental design and metrics
- AnnexC_Aggregates.md - Statistical analysis with charts
- AnnexD_Exemplars.md - Representative test cases
- AnnexE_Calibration.md - Confidence calibration analysis
- AnnexF_Ablations.md - Parameter sensitivity studies
- AnnexG_Latency.md - Performance analysis
- AnnexH_Reproducibility.md - Environment and reproduction info
- Full_Report.html - Complete interactive report

### Exemplars (exemplars/)
- Side-by-side comparison HTML files
- Domain-specific top performers
- exemplars_summary.html - Navigation page

### Ablations (ablations/)
- ablation_summary.json - Detailed results
- ablation_results.csv - Tabulated data
- ablation_report.html - Analysis report

### Reproducibility (configs_snapshot/)
- environment_snapshot.json - Complete environment info
- Original configuration files

## Statistics
- **Total Tests:** {len(primary_results['metrics'])}
- **Success Rate:** {len([m for m in primary_results['metrics'] if m.success]) / len(primary_results['metrics']) * 100:.1f}%
- **Domains:** {len(set(exemplar_files.keys()))}
- **Exemplars:** {sum(len(files) for files in exemplar_files.values())}

## Usage for Patent Filing

1. **Executive Summary** (AnnexA) - Include in patent application claims
2. **Methodology** (AnnexB) - Reference for experimental validation
3. **Aggregates** (AnnexC) - Performance evidence tables
4. **Exemplars** (exemplars/) - Concrete implementation examples
5. **Ablations** (AnnexF) - Demonstrate parameter criticality
6. **Full Report** (Full_Report.html) - Comprehensive evidence

All files are self-contained and ready for legal review.
"""

        readme_file = self.run_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        deliverables['readme'] = readme_file

        return deliverables


def main():
    """Main entry point for patent orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate CEGO Patent Evidence Pack")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                       help="Path to benchmark configuration")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory (default: output/patent_runs)")
    parser.add_argument("--no-ablations", action="store_true",
                       help="Skip ablation studies")
    parser.add_argument("--no-stress", action="store_true",
                       help="Skip stress testing")
    parser.add_argument("--generate-pdfs", action="store_true",
                       help="Generate PDF versions of annexes (requires pandoc)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize orchestrator
    orchestrator = PatentOrchestrator(args.output_dir)

    try:
        # Generate patent evidence pack
        deliverables = orchestrator.generate_full_patent_evidence(
            args.config,
            run_ablations=not args.no_ablations,
            run_stress_test=not args.no_stress,
            generate_pdfs=args.generate_pdfs
        )

        print("\\n" + "="*60)
        print("PATENT EVIDENCE GENERATION COMPLETE")
        print("="*60)
        print(f"Output directory: {orchestrator.run_dir}")
        print(f"Manifest: {deliverables['manifest']}")
        print()
        print("Key deliverables:")
        print(f"  📋 Full Report: {deliverables.get('Full_Report', 'N/A')}")
        print(f"  📁 Patent Pack: {orchestrator.patent_pack_dir}")
        print(f"  🔍 Exemplars: {orchestrator.exemplars_dir}")
        print(f"  ⚗️ Ablations: {orchestrator.ablations_dir}")
        print(f"  📝 Manifest: {deliverables['manifest']}")
        print()
        print("Ready for patent counsel review!")

    except Exception as e:
        logger.error(f"Patent evidence generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()