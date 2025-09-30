import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from api.routes import router
from config.settings import settings
from utils.logging_config import setup_logging, get_logger
from src.model import get_model, cleanup_model

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Track startup time
startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting Text-to-Image AI Service", version=settings.api_version)

    try:
        # Pre-load the model during startup
        logger.info("Loading Stable Diffusion model...")
        model = get_model()
        logger.info("Model loaded successfully")

        # Perform a health check
        if model.health_check():
            logger.info("Model health check passed")
        else:
            logger.warning("Model health check failed")

    except Exception as e:
        logger.error("Failed to initialize model during startup", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("Shutting down Text-to-Image AI Service")
    cleanup_model()
    logger.info("Service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add trusted host middleware for production
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"],  # Configure with actual hosts in production
    )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()

    # Log request
    logger.info(
        "Incoming request",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
    )

    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time,
    )

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        url=str(request.url),
        method=request.method,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "details": {"message": "An unexpected error occurred"},
        },
    )


# Mount static files for web interface
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint - redirect to web interface."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")


@app.get("/api")
async def api_info():
    """API information endpoint."""
    uptime = time.time() - startup_time
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "status": "running",
        "uptime": uptime,
        "environment": settings.environment,
        "web_interface": "/static/index.html",
        "api_docs": "/docs",
        "api_endpoints": {
            "generate": "/api/v1/generate",
            "health": "/api/v1/health",
            "model_info": "/api/v1/model/info"
        }
    }


@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint (for Prometheus scraping)."""
    uptime = time.time() - startup_time
    return {
        "uptime_seconds": uptime,
        "environment": settings.environment,
        "model_name": settings.model_name,
    }


def main():
    """Main entry point for running the application."""
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()