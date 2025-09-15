# CEGO Patent Evidence Pack Makefile

.PHONY: help patent-pack quick-patent full-patent test-basic clean install

# Default target
help:
	@echo "CEGO Patent Evidence Pack Generator"
	@echo "=================================="
	@echo ""
	@echo "Available targets:"
	@echo "  patent-pack     - Generate full patent evidence pack (recommended)"
	@echo "  quick-patent    - Generate patent pack without ablations (faster)"
	@echo "  full-patent     - Generate patent pack with PDFs (requires pandoc)"
	@echo "  test-basic      - Run basic functionality tests"
	@echo "  benchmark       - Run basic benchmark only"
	@echo "  clean          - Clean output directories"
	@echo "  install        - Install Python dependencies"
	@echo ""

# Install dependencies
install:
	pip install -r cego_bench/requirements.txt

# Test basic functionality
test-basic:
	python test_benchmark_basic.py

# Generate full patent evidence pack
patent-pack:
	python generate_patent_pack.py --config cego_bench/configs/default.yaml

# Generate quick patent pack (no ablations)
quick-patent:
	python generate_patent_pack.py --config cego_bench/configs/default.yaml --no-ablations --no-stress

# Generate full patent pack with PDFs
full-patent:
	python generate_patent_pack.py --config cego_bench/configs/default.yaml --generate-pdfs

# Run basic benchmark only
benchmark:
	python -m cego_bench.runners.run_bench --config cego_bench/configs/default.yaml

# Clean output directories
clean:
	rm -rf output/