from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Configuration
    api_title: str = "Text-to-Image AI Service"
    api_version: str = "0.1.0"
    api_description: str = "TensorFlow-based text-to-image generation service"

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Host to bind the server")
    port: int = Field(default=8000, description="Port to bind the server")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload for development")

    # Model Configuration
    model_name: str = Field(default="stabilityai/stable-diffusion-2-1", description="Model to use for generation")
    model_cache_dir: str = Field(default="./models", description="Directory to cache models")
    max_image_size: int = Field(default=512, description="Maximum image size for generation")
    default_steps: int = Field(default=50, description="Default number of diffusion steps")
    max_steps: int = Field(default=100, description="Maximum number of diffusion steps allowed")

    # Performance Configuration
    batch_size: int = Field(default=1, description="Batch size for inference")
    enable_memory_efficient: bool = Field(default=True, description="Enable memory efficient attention")
    enable_xformers: bool = Field(default=False, description="Enable xFormers optimization")

    # GPU Configuration
    gpu_memory_fraction: float = Field(default=0.9, description="Fraction of GPU memory to use")
    mixed_precision: bool = Field(default=True, description="Enable mixed precision training")

    # Security and Rate Limiting
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute per client")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent requests")

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json or text")

    # Cloud Configuration
    cloud_provider: Optional[str] = Field(default=None, description="Cloud provider: aws, gcp, azure")
    storage_bucket: Optional[str] = Field(default=None, description="Cloud storage bucket for models/outputs")

    # Redis Configuration (for caching)
    redis_url: Optional[str] = Field(default=None, description="Redis URL for caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # Health Check Configuration
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")

    # Environment Configuration
    environment: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=False, description="Enable debug mode")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()