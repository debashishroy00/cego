# CEGO Docker Management
# Makefile for easy Docker operations

.PHONY: help build run-dev run-prod test shell benchmark clean lint check

help: ## Show this help message
	@echo "CEGO Docker Commands:"
	@echo "===================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build all Docker images
	@echo "Building CEGO Docker images..."
	docker-compose build

run-dev: ## Run development server with hot reload
	@echo "Starting CEGO development server..."
	docker-compose up cego-dev

run-prod: ## Run production server
	@echo "Starting CEGO production server..."
	docker-compose up cego-prod

test: ## Run full test suite in container
	@echo "Running CEGO test suite..."
	docker build -f Dockerfile.test -t cego-test .
	docker run --rm cego-test

test-watch: ## Run tests with file watching
	@echo "Running tests with file watching..."
	docker-compose run --rm cego-dev pytest --watch

shell: ## Open interactive shell in development container
	@echo "Opening shell in CEGO development container..."
	docker-compose run --rm cego-dev /bin/bash

benchmark: ## Run performance benchmarks
	@echo "Running CEGO benchmarks..."
	docker-compose run --rm cego-dev python tests/benchmarks/validate_reduction.py

lint: ## Run code linting
	@echo "Running code quality checks..."
	docker-compose run --rm cego-dev flake8 src/
	docker-compose run --rm cego-dev black --check src/

check: ## Run all quality checks (tests + lint + complexity)
	@echo "Running comprehensive quality checks..."
	make test
	make lint
	docker-compose run --rm cego-dev radon cc src/ -nb

clean: ## Clean up Docker resources
	@echo "Cleaning up Docker resources..."
	docker-compose down -v
	docker system prune -f

logs-dev: ## Show development server logs
	docker-compose logs -f cego-dev

logs-prod: ## Show production server logs
	docker-compose logs -f cego-prod

health: ## Check service health
	@echo "Checking CEGO service health..."
	curl -s http://localhost:8001/health | python -m json.tool

metrics: ## Get service metrics
	@echo "Fetching CEGO metrics..."
	curl -s http://localhost:8001/metrics | python -m json.tool

# Development workflow shortcuts
dev-setup: build ## Initial development setup
	@echo "CEGO development environment ready!"
	@echo "Run 'make run-dev' to start the development server"

quick-test: ## Quick test run (no container rebuild)
	docker-compose run --rm cego-dev python -m pytest tests/unit/ -v

# CI/CD commands
ci-build: ## CI build (no cache)
	docker build --no-cache -f Dockerfile.prod -t cego:latest .

ci-test: ## CI test run
	docker build -f Dockerfile.test -t cego-test:latest .
	docker run --rm cego-test:latest

# File size check per guidelines
check-files: ## Check file sizes per CEGO guidelines
	@echo "Checking file sizes (target: 300 lines, limit: 500)..."
	@find src -name "*.py" | xargs wc -l | sort -rn | head -10