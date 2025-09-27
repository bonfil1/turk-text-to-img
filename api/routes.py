import time
import asyncio
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import tensorflow as tf
import psutil

from api.models import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    HealthCheckResponse,
    ModelInfoResponse,
    ErrorResponse,
    GeneratedImage,
)
from src.model import get_model, StableDiffusionModel
from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False) if settings.api_key else None


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Verify API key if authentication is enabled."""
    if not settings.api_key:
        return True

    if not credentials or credentials.credentials != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


@router.post(
    "/generate",
    response_model=ImageGenerationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    summary="Generate images from text",
    description="Generate one or more images based on a text prompt using Stable Diffusion",
)
async def generate_images(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_api_key),
) -> ImageGenerationResponse:
    """Generate images from text prompts."""
    start_time = time.time()

    try:
        logger.info("Starting image generation", prompt=request.prompt[:50])

        # Get model instance
        model = get_model()

        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        pil_images = await loop.run_in_executor(
            None,
            model.generate_image,
            request.prompt,
            request.negative_prompt,
            request.num_steps,
            request.guidance_scale,
            request.seed,
            request.batch_size,
        )

        # Convert images to response format
        generated_images: List[GeneratedImage] = []
        for i, pil_image in enumerate(pil_images):
            if request.output_format == "base64":
                image_data = model.image_to_base64(pil_image)
            else:
                # In a real implementation, you'd upload to cloud storage and return URL
                image_data = model.image_to_base64(pil_image)

            generated_images.append(
                GeneratedImage(
                    image_data=image_data,
                    format="PNG",
                    seed=request.seed,
                )
            )

        generation_time = time.time() - start_time

        # Log metrics in background
        background_tasks.add_task(
            log_generation_metrics,
            prompt=request.prompt,
            num_images=len(generated_images),
            generation_time=generation_time,
        )

        logger.info(
            "Image generation completed",
            generation_time=generation_time,
            num_images=len(generated_images),
        )

        return ImageGenerationResponse(
            images=generated_images,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            parameters={
                "num_steps": request.num_steps,
                "guidance_scale": request.guidance_scale,
                "batch_size": request.batch_size,
                "seed": request.seed,
            },
            generation_time=generation_time,
        )

    except Exception as e:
        logger.error("Image generation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Image generation failed",
                error_code="GENERATION_ERROR",
                details={"message": str(e)},
            ).dict(),
        )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Check the health status of the service and model",
)
async def health_check() -> HealthCheckResponse:
    """Perform a health check on the service."""
    try:
        model = get_model()
        model_loaded = model.health_check()

        # Check GPU availability
        gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0

        # Get memory usage
        memory_info = psutil.virtual_memory()
        memory_usage = {
            "total": memory_info.total,
            "available": memory_info.available,
            "percent": memory_info.percent,
        }

        # Calculate uptime (simplified - in production, track actual start time)
        uptime = time.time()  # Placeholder

        status = "healthy" if model_loaded else "unhealthy"

        return HealthCheckResponse(
            status=status,
            model_loaded=model_loaded,
            gpu_available=gpu_available,
            memory_usage=memory_usage,
            uptime=uptime,
        )

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthCheckResponse(
            status="unhealthy",
            model_loaded=False,
            gpu_available=False,
            uptime=0,
        )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get model information",
    description="Get information about the loaded model and its capabilities",
)
async def get_model_info(_: bool = Depends(verify_api_key)) -> ModelInfoResponse:
    """Get information about the loaded model."""
    try:
        model = get_model()
        model_info = model.get_model_info()

        return ModelInfoResponse(
            model_name=model_info["model_name"],
            model_parameters={
                "max_image_size": model_info["max_image_size"],
                "default_steps": model_info["default_steps"],
                "max_steps": model_info["max_steps"],
                "batch_size": model_info["batch_size"],
            },
            capabilities={
                "mixed_precision": model_info["mixed_precision"],
                "memory_efficient": model_info["memory_efficient"],
                "supported_formats": ["PNG", "JPEG"],
                "max_prompt_length": 1000,
            },
        )

    except Exception as e:
        logger.error("Failed to get model info", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Failed to get model information",
                error_code="MODEL_INFO_ERROR",
                details={"message": str(e)},
            ).dict(),
        )


async def log_generation_metrics(prompt: str, num_images: int, generation_time: float) -> None:
    """Log generation metrics for monitoring."""
    logger.info(
        "Generation metrics",
        prompt_length=len(prompt),
        num_images=num_images,
        generation_time=generation_time,
        images_per_second=num_images / generation_time if generation_time > 0 else 0,
    )