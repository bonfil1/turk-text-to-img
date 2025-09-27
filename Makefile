.PHONY: help install install-dev lint format test test-unit test-integration test-cov clean docker-build docker-run docker-dev setup-dev check-env run dev

# Default target
help:
	@echo "Available commands:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  setup-dev      Setup development environment"
	@echo "  lint           Run all linting tools"
	@echo "  format         Format code with black and isort"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-cov       Run tests with coverage report"
	@echo "  clean          Clean up cache and build files"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run application in Docker"
	@echo "  docker-dev     Run development environment with Docker Compose"
	@echo "  run            Run the application locally"
	@echo "  dev            Run development server with auto-reload"
	@echo "  check-env      Check environment setup"

# Environment setup
check-env:
	@echo "Checking environment..."
	@python --version
	@which python
	@pip --version

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

setup-dev: install-dev
	pre-commit install
	@echo "Development environment setup complete!"

# Code quality
lint:
	@echo "Running linting tools..."
	flake8 src api utils config tests
	pylint src api utils config
	mypy src api utils config
	bandit -r src api utils config -x tests/

format:
	@echo "Formatting code..."
	black .
	isort .

# Testing
test:
	@echo "Running all tests..."
	pytest

test-unit:
	@echo "Running unit tests..."
	pytest -m "unit and not slow"

test-integration:
	@echo "Running integration tests..."
	pytest -m "integration"

test-cov:
	@echo "Running tests with coverage..."
	pytest --cov=src --cov=api --cov-report=html --cov-report=term

# Cleanup
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -f docker/Dockerfile -t turk-text-to-img:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 turk-text-to-img:latest

docker-dev:
	@echo "Starting development environment with Docker Compose..."
	docker-compose up --build

docker-stop:
	@echo "Stopping Docker Compose services..."
	docker-compose down

# Application commands
run:
	@echo "Running application..."
	python -m api.main

dev:
	@echo "Starting development server..."
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Pre-commit hooks
pre-commit:
	pre-commit run --all-files

# CI/CD helpers
ci-test: install-dev lint test

ci-build: docker-build

# Model management
download-models:
	@echo "Creating models directory..."
	mkdir -p models
	@echo "Models will be downloaded automatically on first run"

# Database/Cache management
redis-start:
	docker run -d --name turk-redis -p 6379:6379 redis:7-alpine

redis-stop:
	docker stop turk-redis && docker rm turk-redis

# Production deployment helpers
prod-build:
	docker build -f docker/Dockerfile --target production -t turk-text-to-img:prod .

prod-run:
	docker run -d \
		--name turk-text-to-img-prod \
		-p 8000:8000 \
		-e ENVIRONMENT=production \
		-e LOG_LEVEL=INFO \
		turk-text-to-img:prod

# Monitoring
logs:
	docker logs -f turk-text-to-img-prod

# Health checks
health:
	curl -f http://localhost:8000/api/v1/health || echo "Service is not healthy"

# Documentation
docs:
	@echo "Starting documentation server..."
	@echo "API docs available at: http://localhost:8000/docs"
	@echo "ReDoc available at: http://localhost:8000/redoc"