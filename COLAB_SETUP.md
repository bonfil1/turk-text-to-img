# 🚀 Google Colab Deployment Guide

Deploy your text-to-image AI service to Google Colab in 5 minutes with free GPU acceleration!

## 🎯 Quick Setup

### Step 1: Get ngrok Auth Token (Required)

Since ngrok now requires authentication, you need a free account:

1. **Sign up**: Go to [ngrok.com/signup](https://dashboard.ngrok.com/signup) (free account)
2. **Get auth token**: After signup, go to [your authtoken page](https://dashboard.ngrok.com/get-started/your-authtoken)
3. **Copy the token**: It looks like `2abc123_def456ghi789jkl...`

### Step 2: Deploy to Colab

#### Option 1: Upload Notebook (Recommended)

1. **Download the notebook**: Download `colab_notebook.ipynb` from this repository
2. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com/)
3. **Upload notebook**: File → Upload notebook → Select `colab_notebook.ipynb`
4. **Enable GPU**: Runtime → Change runtime type → Hardware accelerator → **GPU**
5. **Add your ngrok token**: In the ngrok setup cell, replace `YOUR_NGROK_AUTH_TOKEN_HERE` with your actual token
6. **Run all cells**: Runtime → Run all (or Ctrl+F9)
7. **Get your URL**: Wait for the ngrok URL to appear (usually takes 5-10 minutes)

### Option 2: Manual Setup

Copy and paste this into a new Colab notebook:

```python
# Cell 1: Check GPU and install dependencies
!nvidia-smi
!pip install -q tensorflow>=2.15.0 keras-cv>=0.6.0 fastapi>=0.104.0
!pip install -q uvicorn[standard] python-multipart pillow pydantic
!pip install -q pydantic-settings python-dotenv structlog nest-asyncio pyngrok

# IMPORTANT: Set your ngrok auth token here
NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN_HERE"  # Replace with your actual token from https://dashboard.ngrok.com/get-started/your-authtoken

# Cell 2: Setup the service
import os
import time
import asyncio
import nest_asyncio
from typing import Optional
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
import threading

# Enable nested asyncio for Colab
nest_asyncio.apply()

# Configure TensorFlow
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"✅ Configured {len(gpus)} GPU(s)")

# Define API models
class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    num_steps: int = Field(25, ge=10, le=50)
    guidance_scale: float = Field(7.5, ge=1.0, le=15.0)
    seed: Optional[int] = Field(None, ge=0)

class ImageResponse(BaseModel):
    image_base64: str
    prompt: str
    generation_time: float

# Load Stable Diffusion model
print("🔄 Loading Stable Diffusion model...")
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=False)
print("✅ Model loaded!")

# Create FastAPI app
app = FastAPI(title="Text-to-Image AI (Colab)", version="1.0.0")

@app.post("/generate", response_model=ImageResponse)
async def generate_image_endpoint(request: ImageRequest):
    start_time = time.time()

    if request.seed is not None:
        tf.random.set_seed(request.seed)
        np.random.seed(request.seed)

    # Generate image
    generated_images = model.text_to_image(
        prompt=request.prompt,
        batch_size=1,
        num_steps=request.num_steps,
        guidance_scale=request.guidance_scale,
    )

    # Convert to PIL and base64
    img_array = generated_images[0]
    img_array = (img_array + 1.0) * 127.5
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(img_array)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return ImageResponse(
        image_base64=img_base64,
        prompt=request.prompt,
        generation_time=time.time() - start_time
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu_available": len(gpus) > 0}

@app.get("/")
async def root():
    return {"message": "Text-to-Image AI (Colab)", "docs_url": "/docs"}

# Cell 3: Start the server
def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Set ngrok auth token and start tunnel
from pyngrok import ngrok
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
public_url = ngrok.connect(8000)
print("\n" + "="*60)
print("🎨 TEXT-TO-IMAGE AI SERVICE IS LIVE!")
print("="*60)
print(f"🌐 Public URL: {public_url}")
print(f"📖 API Docs: {public_url}/docs")
print(f"❤️ Health: {public_url}/health")
print("="*60)

# Start server in background
server_thread = threading.Thread(target=start_server)
server_thread.daemon = True
server_thread.start()

print("✅ Service started! Use the URLs above to access your API.")
```

## 🎨 Using Your Deployed Service

### Via Web Interface
1. Click the ngrok URL from the output
2. Add `/docs` to access interactive API documentation
3. Test the `/generate` endpoint directly in browser

### Via Python Code
```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Replace with your actual ngrok URL
url = "https://your-ngrok-url.ngrok.io/generate"

response = requests.post(url, json={
    "prompt": "a beautiful landscape painting",
    "num_steps": 25,
    "guidance_scale": 7.5
})

if response.status_code == 200:
    result = response.json()
    print(f"Generated in {result['generation_time']:.1f} seconds")

    # Display the image
    img_data = base64.b64decode(result['image_base64'])
    image = Image.open(BytesIO(img_data))
    image.show()
```

### Via cURL
```bash
curl -X POST "https://your-ngrok-url.ngrok.io/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a futuristic city at sunset",
    "num_steps": 30,
    "guidance_scale": 8.0
  }'
```

## ⚙️ Configuration Options

### Performance Settings
- **num_steps**: 10-25 for fast, 25-50 for quality
- **guidance_scale**: 7.5 for balanced, 10+ for strict prompt following
- **GPU**: Enable T4 (free) or A100/V100 (Colab Pro) for best performance

### Colab Runtime Options
- **Standard**: Free T4 GPU, ~12 hours session
- **Colab Pro**: Faster GPUs (A100/V100), longer sessions
- **High-RAM**: Recommended for better performance

## 🔧 Troubleshooting

### Common Issues

**"Model loading takes forever"**
- First run: 5-10 minutes is normal (downloading model weights)
- Subsequent runs: Much faster

**"ngrok URL doesn't work"**
- Wait for model to fully load before accessing
- Check if server started successfully in logs

**"Out of memory errors"**
- Use lower num_steps (10-20)
- Enable High-RAM runtime
- Restart runtime if needed

**"Session disconnected"**
- Colab free tier has 12-hour limit
- Keep browser tab active
- Consider Colab Pro for longer sessions

### Advanced Tips

**Stable URLs** (Optional)
```python
# Get ngrok auth token from: https://dashboard.ngrok.com/get-started/your-authtoken
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_AUTH_TOKEN_HERE")
```

**Custom Model Settings**
```python
# Modify before loading model
model = keras_cv.models.StableDiffusion(
    img_width=768,  # Higher resolution (requires more memory)
    img_height=768,
    jit_compile=False
)
```

## 🚀 What's Next?

After deploying to Colab:

1. **Share your API**: Send the ngrok URL to others
2. **Build a frontend**: Connect to your deployed API
3. **Integrate with apps**: Use the REST API in your projects
4. **Scale up**: Deploy to cloud platforms for production

## 📝 Notes

- **Free Tier**: T4 GPU with usage limits
- **Session Time**: 12 hours maximum (free tier)
- **Public Access**: ngrok URLs are publicly accessible
- **Security**: Don't use for sensitive data without additional security
- **Persistence**: Models and data don't persist between sessions

Perfect for testing, demos, and sharing your AI service with others! 🎨✨