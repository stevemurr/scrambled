.PHONY: help env install install-dev clean test format lint verify-clap

# Default Python interpreter
PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Scrambled Hacks - Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

env: clean ## Create virtual environment and install all dependencies
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip setuptools wheel
	@echo "Installing PyTorch (CPU version for Apple Silicon)..."
	$(PIP) install torch torchvision torchaudio
	@echo "Installing package with dependencies..."
	$(PIP) install -e .
	@echo ""
	@echo "✅ Environment setup complete!"
	@echo ""
	@echo "Activate with: source $(VENV_BIN)/activate"
	@echo "Run 'make verify-clap' to test CLAP model loading"

install: ## Install package in development mode
	@echo "Installing package in development mode..."
	$(PIP) install -e .

install-dev: ## Install package with dev dependencies (black, pytest, etc.)
	@echo "Installing package with dev dependencies..."
	$(PIP) install -e ".[dev]"

verify-clap: ## Test CLAP model loading (downloads ~1GB on first run)
	@echo "Testing CLAP model loading..."
	@$(VENV_BIN)/python -c "from transformers import ClapModel, ClapProcessor; \
		print('Loading CLAP model...'); \
		model = ClapModel.from_pretrained('laion/clap-htsat-unfused'); \
		processor = ClapProcessor.from_pretrained('laion/clap-htsat-unfused'); \
		print('✅ CLAP model loaded successfully!')"

test: ## Run tests with pytest
	@echo "Running tests..."
	$(VENV_BIN)/pytest

format: ## Format code with black and isort
	@echo "Formatting code..."
	$(VENV_BIN)/black src/ tests/
	$(VENV_BIN)/isort src/ tests/

lint: ## Run linters (flake8, mypy)
	@echo "Running linters..."
	$(VENV_BIN)/flake8 src/ tests/
	$(VENV_BIN)/mypy src/

clean: ## Remove virtual environment and cache files
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"
