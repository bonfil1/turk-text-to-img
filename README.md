# Text-to-Image AI Service

A production-ready text-to-image generation service built with TensorFlow, KerasCV, and FastAPI. This service provides a REST API for generating high-quality images from text prompts using Stable Diffusion.

## Features

- 🎨 **High-Quality Image Generation**: Powered by Stable Diffusion via KerasCV
- 🚀 **Fast API**: RESTful API built with FastAPI for high performance
- 🐳 **Docker Ready**: Containerized for easy deployment
- ☁️ **Cloud Optimized**: Ready for deployment on AWS, GCP, or Azure
- 🔧 **Production Ready**: Comprehensive logging, monitoring, and error handling
- 🧪 **Well Tested**: Extensive test suite with unit and integration tests
- 📊 **Monitoring**: Built-in health checks and metrics
- 🔒 **Secure**: Optional API key authentication and security best practices

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd turk-text-to-img

# Start the service with Docker Compose
docker-compose up --build

# The API will be available at http://localhost:8000
```

### Local Development

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
make install-dev

# Setup development environment
make setup-dev

# Run the development server
make dev
```

## API Usage

### Generate Images

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

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### API Documentation

- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## Configuration

The service can be configured using environment variables. Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
```

Key configuration options:

- `MODEL_NAME`: Stable Diffusion model to use
- `MAX_IMAGE_SIZE`: Maximum image resolution
- `API_KEY`: Optional API key for authentication
- `REDIS_URL`: Redis URL for caching (optional)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Development

### Available Commands

```bash
make help                 # Show all available commands
make setup-dev           # Setup development environment
make lint                # Run linting tools
make format              # Format code
make test                # Run all tests
make test-unit           # Run unit tests only
make docker-build        # Build Docker image
```

### Code Quality

This project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Static type checking
- **pytest**: Testing framework
- **pre-commit**: Git hooks for quality checks

### Running Tests

```bash
# Run all tests
make test

# Run only unit tests
make test-unit

# Run with coverage
make test-cov
```

## Deployment

### Docker Deployment

```bash
# Build production image
make prod-build

# Run in production mode
make prod-run
```

### Cloud Deployment

The service is designed to be cloud-ready with:

- Environment-based configuration
- Health checks for load balancers
- Metrics endpoints for monitoring
- Structured logging
- Graceful shutdown handling

### Environment Variables for Production

```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
API_KEY=your-secure-api-key
REDIS_URL=redis://your-redis-instance:6379/0
MODEL_CACHE_DIR=/app/models
```

## Monitoring

### Health Endpoints

- `GET /api/v1/health` - Service health check
- `GET /metrics` - Basic metrics for monitoring
- `GET /` - Service status

### Logging

The service uses structured logging (JSON format) suitable for log aggregation systems like ELK stack or cloud logging services.

## Performance Optimization

### GPU Support

For GPU acceleration, use the GPU-enabled Docker image:

```bash
# Uncomment the GPU service in docker-compose.yml
docker-compose up text-to-image-gpu
```

### Memory Management

The service includes several memory optimization features:

- Memory-efficient attention mechanisms
- Model caching and reuse
- Configurable batch sizes
- Automatic cleanup

## Security

- Optional API key authentication
- Input validation and sanitization
- Rate limiting (configurable)
- Secure container configuration
- No exposure of sensitive information in logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting: `make ci-test`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure sufficient disk space and memory
2. **GPU Not Detected**: Verify CUDA installation and Docker GPU support
3. **Port Already in Use**: Change the port in docker-compose.yml or .env

### Support

For issues and questions:

1. Check the [documentation](http://localhost:8000/docs)
2. Review the logs: `make logs`
3. Create an issue on GitHub

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Model Layer   │    │   TensorFlow    │
│   REST API      │───▶│   (src/model)   │───▶│   + KerasCV     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Validation    │    │   Logging &     │    │   Monitoring    │
│   & Auth        │    │   Config        │    │   & Health      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```