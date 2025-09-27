from typing import Optional, List, Literal
from pydantic import BaseModel, Field, validator


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the desired image", min_length=1, max_length=1000)
    negative_prompt: Optional[str] = Field(None, description="Text description of what to avoid", max_length=1000)
    num_steps: Optional[int] = Field(50, description="Number of diffusion steps", ge=1, le=100)
    guidance_scale: float = Field(7.5, description="How closely to follow the prompt", ge=1.0, le=20.0)
    seed: Optional[int] = Field(None, description="Random seed for reproducible results", ge=0)
    batch_size: int = Field(1, description="Number of images to generate", ge=1, le=4)
    output_format: Literal["base64", "url"] = Field("base64", description="Output format for images")

    @validator("prompt")
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()

    @validator("negative_prompt")
    def validate_negative_prompt(cls, v):
        if v is not None:
            return v.strip()
        return v


class GeneratedImage(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image or URL")
    format: str = Field("PNG", description="Image format")
    seed: Optional[int] = Field(None, description="Seed used for generation")


class ImageGenerationResponse(BaseModel):
    images: List[GeneratedImage] = Field(..., description="Generated images")
    prompt: str = Field(..., description="Original prompt used")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt used")
    parameters: dict = Field(..., description="Generation parameters used")
    generation_time: float = Field(..., description="Time taken to generate images in seconds")


class HealthCheckResponse(BaseModel):
    status: Literal["healthy", "unhealthy"] = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_usage: Optional[dict] = Field(None, description="Memory usage information")
    uptime: float = Field(..., description="Service uptime in seconds")


class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., description="Name of the loaded model")
    model_parameters: dict = Field(..., description="Model configuration parameters")
    capabilities: dict = Field(..., description="Model capabilities and limits")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[dict] = Field(None, description="Additional error details")


class GenerationStatus(BaseModel):
    status: Literal["queued", "processing", "completed", "failed"] = Field(..., description="Generation status")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    estimated_completion: Optional[float] = Field(None, description="Estimated completion time in seconds")
    message: Optional[str] = Field(None, description="Status message")