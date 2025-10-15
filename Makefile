.PHONY: env install install-dev clean test format lint help

# Default Python interpreter
PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip

help:
	@echo "Scrambled Hacks - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make env          Create virtual environment and install dependencies"
	@echo "  make install      Install package in development mode"
	@echo "  make install-dev  Install package with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test         Run tests with pytest"
	@echo "  make format       Format code with black and isort"
	@echo "  make lint         Run linters (flake8, mypy)"
	@echo "  make clean        Remove virtual environment and cache files"
	@echo ""
	@echo "Verification:"
	@echo "  make verify-clap  Test CLAP model loading"
	@echo ""

env: clean
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

install:
	@echo "Installing package in development mode..."
	$(PIP) install -e .

install-dev:
	@echo "Installing package with dev dependencies..."
	$(PIP) install -e ".[dev]"

verify-clap:
	@echo "Testing CLAP model loading..."
	@$(VENV_BIN)/python -c "from transformers import ClapModel, ClapProcessor; \
		print('Loading CLAP model...'); \
		model = ClapModel.from_pretrained('laion/clap-htsat-unfused'); \
		processor = ClapProcessor.from_pretrained('laion/clap-htsat-unfused'); \
		print('✅ CLAP model loaded successfully!')"

test:
	@echo "Running tests..."
	$(VENV_BIN)/pytest

format:
	@echo "Formatting code..."
	$(VENV_BIN)/black src/ tests/
	$(VENV_BIN)/isort src/ tests/

lint:
	@echo "Running linters..."
	$(VENV_BIN)/flake8 src/ tests/
	$(VENV_BIN)/mypy src/

clean:
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
