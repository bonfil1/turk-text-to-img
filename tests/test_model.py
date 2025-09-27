import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import tensorflow as tf

from src.model import StableDiffusionModel, get_model, cleanup_model


class TestStableDiffusionModel:
    """Test cases for StableDiffusionModel class."""

    @pytest.fixture
    def mock_keras_cv(self):
        """Mock keras_cv for testing."""
        with patch("src.model.keras_cv") as mock:
            mock_model = Mock()
            mock_model.text_to_image.return_value = [
                np.random.rand(512, 512, 3) * 2 - 1  # Random image array in [-1, 1]
            ]
            mock.models.StableDiffusion.return_value = mock_model
            yield mock

    @pytest.fixture
    def mock_tf_config(self):
        """Mock TensorFlow GPU configuration."""
        with patch("src.model.tf.config.experimental.list_physical_devices") as mock_gpus, \
             patch("src.model.tf.config.experimental.set_memory_growth") as mock_memory, \
             patch("src.model.tf.keras.mixed_precision.set_global_policy") as mock_policy:
            mock_gpus.return_value = [Mock()]  # Simulate one GPU
            yield mock_gpus, mock_memory, mock_policy

    def test_model_initialization(self, mock_keras_cv, mock_tf_config):
        """Test model initialization with default settings."""
        model = StableDiffusionModel()

        assert model.model_name == "stabilityai/stable-diffusion-2-1"
        assert model.model is not None
        mock_keras_cv.models.StableDiffusion.assert_called_once()

    def test_gpu_setup(self, mock_keras_cv, mock_tf_config):
        """Test GPU configuration setup."""
        mock_gpus, mock_memory, mock_policy = mock_tf_config

        model = StableDiffusionModel()

        mock_gpus.assert_called_once_with('GPU')
        mock_memory.assert_called_once()
        mock_policy.assert_called_once_with('mixed_float16')

    def test_generate_image_single(self, mock_keras_cv, mock_tf_config):
        """Test single image generation."""
        model = StableDiffusionModel()

        images = model.generate_image("a cat sitting on a table")

        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        assert images[0].size == (512, 512)

    def test_generate_image_batch(self, mock_keras_cv, mock_tf_config):
        """Test batch image generation."""
        # Mock multiple images
        mock_keras_cv.models.StableDiffusion.return_value.text_to_image.return_value = [
            np.random.rand(512, 512, 3) * 2 - 1,
            np.random.rand(512, 512, 3) * 2 - 1,
        ]

        model = StableDiffusionModel()

        images = model.generate_image("a cat", batch_size=2)

        assert len(images) == 2
        for img in images:
            assert isinstance(img, Image.Image)

    def test_generate_image_with_negative_prompt(self, mock_keras_cv, mock_tf_config):
        """Test image generation with negative prompt."""
        model = StableDiffusionModel()

        images = model.generate_image(
            prompt="a beautiful landscape",
            negative_prompt="blurry, low quality"
        )

        assert len(images) == 1
        # Verify negative prompt was passed to the model
        call_args = mock_keras_cv.models.StableDiffusion.return_value.text_to_image.call_args
        assert call_args[1]["negative_prompt"] == "blurry, low quality"

    def test_generate_image_with_seed(self, mock_keras_cv, mock_tf_config):
        """Test image generation with fixed seed."""
        with patch("src.model.tf.random.set_seed") as mock_tf_seed, \
             patch("src.model.np.random.seed") as mock_np_seed:

            model = StableDiffusionModel()

            images = model.generate_image("a cat", seed=42)

            mock_tf_seed.assert_called_once_with(42)
            mock_np_seed.assert_called_once_with(42)
            assert len(images) == 1

    def test_image_to_base64(self, mock_keras_cv, mock_tf_config):
        """Test image to base64 conversion."""
        model = StableDiffusionModel()

        # Create a test image
        test_image = Image.new("RGB", (100, 100), color="red")

        base64_str = model.image_to_base64(test_image)

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

    def test_get_model_info(self, mock_keras_cv, mock_tf_config):
        """Test getting model information."""
        model = StableDiffusionModel()

        info = model.get_model_info()

        assert "model_name" in info
        assert "max_image_size" in info
        assert "default_steps" in info
        assert info["model_name"] == "stabilityai/stable-diffusion-2-1"

    def test_health_check_success(self, mock_keras_cv, mock_tf_config):
        """Test successful health check."""
        model = StableDiffusionModel()

        result = model.health_check()

        assert result is True

    def test_health_check_failure(self, mock_keras_cv, mock_tf_config):
        """Test health check failure."""
        # Make the model raise an exception
        mock_keras_cv.models.StableDiffusion.return_value.text_to_image.side_effect = Exception("Model error")

        model = StableDiffusionModel()

        result = model.health_check()

        assert result is False

    def test_parameter_validation(self, mock_keras_cv, mock_tf_config):
        """Test parameter validation in generate_image."""
        model = StableDiffusionModel()

        # Test step limit enforcement
        images = model.generate_image("test", num_steps=150)  # Over max

        call_args = mock_keras_cv.models.StableDiffusion.return_value.text_to_image.call_args
        assert call_args[1]["num_steps"] <= 100  # Should be capped at max_steps

    def test_model_not_loaded_error(self, mock_keras_cv, mock_tf_config):
        """Test error when model is not loaded."""
        model = StableDiffusionModel()
        model.model = None  # Simulate unloaded model

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.generate_image("test prompt")


class TestModelSingleton:
    """Test cases for model singleton functionality."""

    def test_get_model_singleton(self):
        """Test that get_model returns the same instance."""
        with patch("src.model.StableDiffusionModel") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            model1 = get_model()
            model2 = get_model()

            assert model1 is model2
            mock_class.assert_called_once()

    def test_cleanup_model(self):
        """Test model cleanup functionality."""
        with patch("src.model.StableDiffusionModel") as mock_class, \
             patch("src.model.tf.keras.backend.clear_session") as mock_clear:

            mock_instance = Mock()
            mock_class.return_value = mock_instance

            # Get model instance
            get_model()

            # Cleanup
            cleanup_model()

            mock_clear.assert_called_once()

    def test_model_recreation_after_cleanup(self):
        """Test that model is recreated after cleanup."""
        with patch("src.model.StableDiffusionModel") as mock_class:
            mock_instance1 = Mock()
            mock_instance2 = Mock()
            mock_class.side_effect = [mock_instance1, mock_instance2]

            model1 = get_model()
            cleanup_model()
            model2 = get_model()

            assert model1 is not model2
            assert mock_class.call_count == 2