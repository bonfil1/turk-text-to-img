import pytest
import tempfile
import os
from unittest.mock import patch

from config.settings import Settings


@pytest.fixture(scope="session")
def temp_model_dir():
    """Create a temporary directory for model storage during tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="function")
def test_settings(temp_model_dir):
    """Create test settings with temporary directories."""
    test_settings = Settings(
        model_cache_dir=temp_model_dir,
        log_level="DEBUG",
        environment="test",
        debug=True,
        api_key=None,  # Disable auth for tests
        max_steps=50,  # Lower for faster tests
        default_steps=10,
        batch_size=1,
        gpu_memory_fraction=0.5,
    )

    with patch("config.settings.settings", test_settings):
        yield test_settings


@pytest.fixture(scope="function")
def mock_tensorflow():
    """Mock TensorFlow operations for testing."""
    with patch("tensorflow.config.experimental.list_physical_devices") as mock_devices, \
         patch("tensorflow.config.experimental.set_memory_growth") as mock_memory, \
         patch("tensorflow.keras.mixed_precision.set_global_policy") as mock_policy, \
         patch("tensorflow.random.set_seed") as mock_tf_seed, \
         patch("tensorflow.keras.backend.clear_session") as mock_clear:

        mock_devices.return_value = []  # No GPUs by default
        yield {
            "devices": mock_devices,
            "memory": mock_memory,
            "policy": mock_policy,
            "seed": mock_tf_seed,
            "clear": mock_clear,
        }


@pytest.fixture(scope="function")
def mock_keras_cv():
    """Mock KerasCV for testing."""
    with patch("keras_cv.models.StableDiffusion") as mock_sd:
        mock_model_instance = mock_sd.return_value
        mock_model_instance.text_to_image.return_value = []
        yield mock_sd


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    test_env_vars = {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "DEBUG",
        "MODEL_CACHE_DIR": "/tmp/test_models",
        "API_KEY": "",
        "REDIS_URL": "",
    }

    original_env = {}
    for key, value in test_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield

    # Restore original environment
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="function")
def sample_image_data():
    """Provide sample image data for testing."""
    import numpy as np
    from PIL import Image
    import io
    import base64

    # Create a sample image
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    pil_image = Image.fromarray(img_array)

    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode()

    return {
        "pil_image": pil_image,
        "numpy_array": img_array,
        "base64_string": base64_str,
    }


@pytest.fixture(scope="function")
def mock_redis():
    """Mock Redis for caching tests."""
    with patch("redis.Redis") as mock_redis_class:
        mock_redis_instance = mock_redis_class.return_value
        mock_redis_instance.get.return_value = None
        mock_redis_instance.set.return_value = True
        mock_redis_instance.exists.return_value = False
        yield mock_redis_instance


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

        # Mark slow tests
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)