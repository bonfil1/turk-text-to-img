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

### 🚀 Google Colab (Fastest - 5 minutes)

Deploy instantly with free GPU acceleration:

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `colab_notebook.ipynb` from this repository
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run all cells
5. Get your public API URL instantly!

**Perfect for**: Testing, demos, sharing with others

### 🐳 Using Docker (Recommended for Local)

```bash
# Clone the repository
git clone <repository-url>
cd turk-text-to-img

# Start the service with Docker Compose
docker-compose up --build

# The API will be available at http://localhost:8000
```

### 💻 Local Development

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

### Google Colab Deployment (Recommended for Testing)

Deploy instantly to Google Colab for free GPU-accelerated text-to-image generation:

#### Quick Start (5 minutes)

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com/)

2. **Enable GPU**: Runtime → Change runtime type → Hardware accelerator → **GPU**

3. **Upload the notebook**: Upload `colab_notebook.ipynb` from this repository

4. **Run all cells**: The notebook will automatically:
   - ✅ Install all dependencies
   - ✅ Load Stable Diffusion model
   - ✅ Start FastAPI server
   - ✅ Create public URL with ngrok

5. **Get your public URL**: After running, you'll get a public URL like `https://abc123.ngrok.io`

#### Manual Setup in Colab

If you prefer to set up manually, run this in a Colab cell:

```python
# Quick setup for Google Colab
!git clone https://github.com/yourusername/turk-text-to-img.git
%cd turk-text-to-img

# Install dependencies
!pip install -q tensorflow>=2.15.0 keras-cv>=0.6.0 fastapi>=0.104.0
!pip install -q uvicorn[standard] python-multipart pillow pydantic
!pip install -q pydantic-settings python-dotenv structlog nest-asyncio pyngrok

# Run the Colab setup
exec(open('colab_setup.py').read())
```

#### What You Get

- **🎨 Public API**: Accessible worldwide via ngrok URL
- **📖 Interactive Docs**: `your-url/docs` for testing
- **⚡ GPU Acceleration**: Free T4 GPU or paid A100/V100
- **🚀 Ready in 5 minutes**: No local setup required

#### Example Usage

Once deployed, test your API:

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Replace with your ngrok URL
url = "https://your-ngrok-url.ngrok.io/generate"

response = requests.post(url, json={
    "prompt": "a beautiful sunset over mountains",
    "num_steps": 25,  # Lower = faster
    "guidance_scale": 7.5
})

if response.status_code == 200:
    result = response.json()
    img_data = base64.b64decode(result['image_base64'])
    image = Image.open(BytesIO(img_data))
    image.show()
```

#### Colab Tips

- **Performance**: Use GPU runtime for 10x faster generation
- **Reliability**: Add ngrok auth token for stable URLs
- **Cost**: Free T4 GPU, upgrade to Colab Pro for A100/V100
- **Limitations**: 12-hour session timeout, usage limits on free tier

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

#### Local Development
1. **Model Loading Errors**: Ensure sufficient disk space and memory
2. **GPU Not Detected**: Verify CUDA installation and Docker GPU support
3. **Port Already in Use**: Change the port in docker-compose.yml or .env

#### Google Colab
1. **ngrok authentication error**:
   - Sign up for free at https://dashboard.ngrok.com/signup
   - Get your token at https://dashboard.ngrok.com/get-started/your-authtoken
   - Replace `YOUR_NGROK_AUTH_TOKEN_HERE` in the notebook
2. **Model loading takes forever**: Wait 5-10 minutes on first run (downloading weights)
3. **Out of memory**: Enable High-RAM runtime or reduce `num_steps`
4. **Session timeout**: Colab free tier has 12-hour limit
5. **"guidance_scale" parameter error**: Automatically handled by fallback to `unconditional_guidance_scale`

### Debug Commands
```bash
# Check service health
curl http://localhost:8000/api/v1/health

# View logs
make logs

# Check environment
make check-env
```

### Quick Fixes
```python
# For Colab ngrok auth error, run this:
from pyngrok import ngrok
ngrok.set_auth_token("your_actual_token_here")
```

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