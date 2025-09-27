# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-ready text-to-image AI service built with TensorFlow, KerasCV, and FastAPI. It provides a REST API for generating high-quality images from text prompts using Stable Diffusion.

## Environment Setup

- **Python Version**: 3.9.4
- **Virtual Environment**: `.venv/` (already created)
- **Package Manager**: pip
- **Framework**: TensorFlow + KerasCV for AI, FastAPI for API

### Quick Start Commands
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
make install-dev

# Setup development environment
make setup-dev

# Run development server
make dev
```

## Common Development Commands

### Development Workflow
```bash
# Run the application locally
make run

# Start development server with auto-reload
make dev

# Build and run with Docker
make docker-build
make docker-run

# Start full development environment
docker-compose up --build
```

### Code Quality and Testing
```bash
# Run all linting tools
make lint

# Format code (black + isort)
make format

# Run all tests
make test

# Run unit tests only
make test-unit

# Run tests with coverage
make test-cov

# Run pre-commit hooks
make pre-commit
```

### Docker Commands
```bash
# Build Docker image
make docker-build

# Run Docker container
make docker-run

# Start development environment with Docker Compose
make docker-dev

# Build production image
make prod-build
```

## Project Structure

```
├── api/                 # FastAPI application
│   ├── main.py         # Application entry point
│   ├── routes.py       # API endpoints
│   └── models.py       # Pydantic models
├── src/                # Core application logic
│   └── model.py        # Stable Diffusion model wrapper
├── config/             # Configuration management
│   └── settings.py     # Environment-based settings
├── utils/              # Utility functions
│   └── logging_config.py
├── tests/              # Test suite
├── docker/             # Docker configuration
└── models/             # Model cache directory
```

## API Endpoints

### Main Endpoints
- `POST /api/v1/generate` - Generate images from text
- `GET /api/v1/health` - Health check
- `GET /api/v1/model/info` - Model information
- `GET /docs` - Interactive API documentation
- `GET /metrics` - Basic metrics

### Example API Usage
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains",
    "num_steps": 50,
    "guidance_scale": 7.5,
    "batch_size": 1
  }'
```

## Configuration

Environment variables can be set in `.env` file (copy from `.env.example`):

### Key Settings
- `MODEL_NAME`: Stable Diffusion model to use
- `MAX_IMAGE_SIZE`: Maximum image resolution (default: 512)
- `API_KEY`: Optional API authentication
- `REDIS_URL`: Redis for caching (optional)
- `LOG_LEVEL`: Logging verbosity
- `ENVIRONMENT`: development/staging/production

## Testing Strategy

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test API endpoints and model integration
- **Mocking**: TensorFlow and KerasCV are mocked in tests for speed
- **Coverage**: Aim for >80% code coverage

### Running Specific Tests
```bash
# Fast unit tests only
pytest -m "unit and not slow"

# Integration tests
pytest -m "integration"

# All tests with coverage
pytest --cov=src --cov=api
```

## Performance Considerations

### GPU Support
- Automatic GPU detection and configuration
- Memory growth enabled to prevent GPU memory issues
- Mixed precision support for better performance

### Memory Optimization
- Model singleton pattern to avoid reloading
- Efficient image processing with PIL
- Configurable batch sizes

### Production Deployment
- Multi-stage Docker builds for smaller images
- Health checks for load balancers
- Structured logging for monitoring
- Graceful shutdown handling

## Cloud Deployment Ready

The service includes:
- Environment-based configuration
- Health check endpoints
- Metrics for monitoring
- Docker containerization
- Security best practices
- Horizontal scaling support

## Code Quality Tools

- **Black**: Code formatting (88 char line length)
- **isort**: Import sorting
- **flake8**: Linting with custom rules
- **mypy**: Static type checking
- **pytest**: Testing framework
- **pre-commit**: Git hooks for quality

## Troubleshooting

### Common Issues
1. **Model loading errors**: Check disk space and memory
2. **GPU not detected**: Verify CUDA installation
3. **Port conflicts**: Change port in docker-compose.yml
4. **Memory issues**: Reduce batch_size or max_image_size

### Debug Commands
```bash
# Check service health
curl http://localhost:8000/api/v1/health

# View logs
make logs

# Check environment
make check-env
```