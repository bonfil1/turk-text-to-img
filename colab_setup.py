"""
Google Colab setup script for the text-to-image AI service.
Run this in a Colab notebook to set up the environment and start the service.
"""

import os
import subprocess
import sys
from pathlib import Path

def install_system_dependencies():
    """Install system dependencies in Colab."""
    print("Installing system dependencies...")

    # Update package list and install required packages
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run([
        "apt-get", "install", "-y", "-qq",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1"
    ], check=True)

    print("✓ System dependencies installed")

def install_python_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")

    # Core dependencies for Colab
    dependencies = [
        "tensorflow>=2.15.0",
        "keras-cv>=0.6.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "python-dotenv>=1.0.0",
        "structlog>=23.2.0",
        "nest-asyncio",  # Required for running async in Colab
        "pyngrok",  # For public URL tunneling
    ]

    for dep in dependencies:
        print(f"Installing {dep}...")
        subprocess.run([sys.executable, "-m", "pip", "install", dep],
                      check=True, capture_output=True)

    print("✓ Python dependencies installed")

def setup_colab_environment():
    """Set up the Colab environment."""
    print("Setting up Colab environment...")

    # Create necessary directories
    os.makedirs("/content/models", exist_ok=True)
    os.makedirs("/content/logs", exist_ok=True)

    # Set environment variables for Colab
    os.environ.update({
        "ENVIRONMENT": "colab",
        "LOG_LEVEL": "INFO",
        "MODEL_CACHE_DIR": "/content/models",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "DEBUG": "true",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "CUDA_VISIBLE_DEVICES": "0" if "COLAB_GPU" in os.environ else "",
    })

    print("✓ Colab environment configured")

def create_colab_app():
    """Create a simplified app for Colab."""

    app_code = '''
import os
import time
import asyncio
import nest_asyncio
from typing import Optional, List
import tensorflow as tf
import keras_cv
import numpy as np
from PIL import Image
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from pyngrok import ngrok

# Enable nested asyncio for Colab
nest_asyncio.apply()

# Configure TensorFlow for Colab
def setup_tensorflow():
    """Configure TensorFlow for Colab environment."""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print(f"✓ Configured {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("⚠️ No GPU detected, using CPU")

class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    num_steps: int = Field(25, ge=10, le=50)  # Reduced for Colab
    guidance_scale: float = Field(7.5, ge=1.0, le=15.0)
    seed: Optional[int] = Field(None, ge=0)

class ImageResponse(BaseModel):
    image_base64: str
    prompt: str
    generation_time: float

class ColabTextToImageModel:
    """Simplified model class for Colab."""

    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load Stable Diffusion model."""
        print("Loading Stable Diffusion model...")
        setup_tensorflow()

        try:
            self.model = keras_cv.models.StableDiffusion(
                img_width=512,
                img_height=512,
                jit_compile=False,  # Disable for Colab compatibility
            )
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise

    def generate_image(self, prompt: str, num_steps: int = 25,
                      guidance_scale: float = 7.5, seed: Optional[int] = None):
        """Generate image from prompt."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        try:
            print(f"Generating image for: {prompt[:50]}...")

            generated_images = self.model.text_to_image(
                prompt=prompt,
                batch_size=1,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
            )

            # Convert to PIL Image
            img_array = generated_images[0]
            img_array = (img_array + 1.0) * 127.5
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)

            return pil_image

        except Exception as e:
            print(f"Generation failed: {e}")
            raise

# Initialize model (global instance)
model_instance = None

def get_model():
    global model_instance
    if model_instance is None:
        model_instance = ColabTextToImageModel()
    return model_instance

# FastAPI app
app = FastAPI(
    title="Text-to-Image AI (Colab)",
    description="Stable Diffusion text-to-image generation in Google Colab",
    version="1.0.0"
)

@app.post("/generate", response_model=ImageResponse)
async def generate_image_endpoint(request: ImageRequest):
    """Generate image from text prompt."""
    start_time = time.time()

    try:
        model = get_model()

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        pil_image = await loop.run_in_executor(
            None,
            model.generate_image,
            request.prompt,
            request.num_steps,
            request.guidance_scale,
            request.seed,
        )

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        generation_time = time.time() - start_time

        return ImageResponse(
            image_base64=img_base64,
            prompt=request.prompt,
            generation_time=generation_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": model.model is not None,
            "gpu_available": len(tf.config.experimental.list_physical_devices("GPU")) > 0
        }
    except:
        return {"status": "unhealthy", "model_loaded": False, "gpu_available": False}

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Text-to-Image AI Service (Colab)",
        "docs_url": "/docs",
        "health_url": "/health"
    }

def start_server_with_ngrok(port=8000):
    """Start the server with ngrok tunnel for public access."""
    print("Starting server with ngrok...")

    # Start ngrok tunnel
    public_url = ngrok.connect(port)
    print(f"\\n🌐 Public URL: {public_url}")
    print(f"📖 API Docs: {public_url}/docs")
    print(f"❤️ Health Check: {public_url}/health")

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    start_server_with_ngrok()
'''

    with open("/content/colab_app.py", "w") as f:
        f.write(app_code)

    print("✓ Colab app created at /content/colab_app.py")

def main():
    """Main setup function for Colab."""
    print("🚀 Setting up Text-to-Image AI Service for Google Colab...")
    print("=" * 60)

    try:
        install_system_dependencies()
        install_python_dependencies()
        setup_colab_environment()
        create_colab_app()

        print("=" * 60)
        print("✅ Setup complete!")
        print("\nNext steps:")
        print("1. Run: exec(open('/content/colab_app.py').read())")
        print("2. Or run: python /content/colab_app.py")
        print("3. Access the public URL provided by ngrok")

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        raise

if __name__ == "__main__":
    main()