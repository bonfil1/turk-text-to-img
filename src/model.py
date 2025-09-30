import os
import logging
from typing import Optional, List, Union
import tensorflow as tf
import keras_cv
import numpy as np
from PIL import Image
import io
import base64

from config.settings import settings

logger = logging.getLogger(__name__)


class StableDiffusionModel:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.model_name
        self.model = None
        self._setup_gpu()
        self._load_model()

    def _setup_gpu(self) -> None:
        """Configure GPU settings for optimal performance."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                if settings.mixed_precision:
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')

                logger.info(f"Configured {len(gpus)} GPU(s) with memory growth enabled")
            except RuntimeError as e:
                logger.error(f"GPU configuration failed: {e}")
        else:
            logger.warning("No GPUs found, using CPU")

    def _load_model(self) -> None:
        """Load the Stable Diffusion model with KerasCV."""
        try:
            os.makedirs(settings.model_cache_dir, exist_ok=True)

            # Load the Stable Diffusion model
            self.model = keras_cv.models.StableDiffusion(
                img_width=settings.max_image_size,
                img_height=settings.max_image_size,
                jit_compile=True,  # Enable XLA compilation for better performance
            )

            logger.info(f"Successfully loaded Stable Diffusion model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_steps: Optional[int] = None,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        batch_size: int = 1,
    ) -> List[Image.Image]:
        """
        Generate images from text prompts.

        Args:
            prompt: Text description of the desired image
            negative_prompt: Text description of what to avoid in the image
            num_steps: Number of diffusion steps (default from settings)
            guidance_scale: How closely to follow the prompt (higher = more faithful)
            seed: Random seed for reproducible results
            batch_size: Number of images to generate

        Returns:
            List of PIL Images
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Validate parameters
        num_steps = num_steps or settings.default_steps
        num_steps = min(num_steps, settings.max_steps)
        batch_size = min(batch_size, settings.batch_size)

        try:
            # Set random seed if provided
            if seed is not None:
                tf.random.set_seed(seed)
                np.random.seed(seed)

            # Generate images
            logger.info(f"Generating {batch_size} image(s) with prompt: '{prompt[:50]}...'")

            # Generate images with version compatibility
            try:
                # Try with guidance_scale first (newer KerasCV versions)
                generated_images = self.model.text_to_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    batch_size=batch_size,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                )
            except TypeError as e:
                if "guidance_scale" in str(e):
                    logger.warning("Using unconditional_guidance_scale for older KerasCV version")
                    # Fallback for older versions that use unconditional_guidance_scale
                    try:
                        generated_images = self.model.text_to_image(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            batch_size=batch_size,
                            num_steps=num_steps,
                            unconditional_guidance_scale=guidance_scale,
                        )
                    except TypeError:
                        # If still failing, try without guidance parameter
                        logger.warning("Using basic parameters without guidance scale")
                        generated_images = self.model.text_to_image(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            batch_size=batch_size,
                            num_steps=num_steps,
                        )
                else:
                    raise e

            # Convert to PIL Images
            pil_images = []
            for img_array in generated_images:
                # Convert from [-1, 1] to [0, 255]
                img_array = (img_array + 1.0) * 127.5
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(img_array)
                pil_images.append(pil_image)

            logger.info(f"Successfully generated {len(pil_images)} image(s)")
            return pil_images

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise

    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "max_image_size": settings.max_image_size,
            "default_steps": settings.default_steps,
            "max_steps": settings.max_steps,
            "batch_size": settings.batch_size,
            "mixed_precision": settings.mixed_precision,
            "memory_efficient": settings.enable_memory_efficient,
        }

    def health_check(self) -> bool:
        """Perform a health check on the model."""
        try:
            if self.model is None:
                return False

            # Simple test generation
            test_prompt = "test"
            test_image = self.model.text_to_image(
                prompt=test_prompt,
                batch_size=1,
                num_steps=1,  # Minimal steps for quick check
            )
            return len(test_image) > 0

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Global model instance
_model_instance: Optional[StableDiffusionModel] = None


def get_model() -> StableDiffusionModel:
    """Get the global model instance, creating it if necessary."""
    global _model_instance
    if _model_instance is None:
        _model_instance = StableDiffusionModel()
    return _model_instance


def cleanup_model() -> None:
    """Clean up the global model instance."""
    global _model_instance
    if _model_instance is not None:
        del _model_instance
        _model_instance = None
        tf.keras.backend.clear_session()
        logger.info("Model cleaned up")