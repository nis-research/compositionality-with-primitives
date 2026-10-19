PY ?= python3.12
TARGET_VERSION ?= py312

.PHONY: help clean lint

help: ## Show this help message
	@echo "Available commands:"
	@echo ""
	@echo "  clean   - Remove Python compiled files, caches, and build artifacts"
	@echo "  lint    - Run black formatting check on source, tests, and top-level .py files"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Usage: make <command>"

clean: ## Remove Python compiled files, caches, and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	@echo "Cleaned."

lint: ## Run black formatting check on source, tests, and top-level .py files
	black --target-version $(TARGET_VERSION) gde *.py
