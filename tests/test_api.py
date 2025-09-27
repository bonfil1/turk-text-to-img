import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from PIL import Image
import io
import base64

from api.main import app
from api.models import ImageGenerationRequest


class TestImageGenerationAPI:
    """Test cases for image generation API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_model(self):
        """Mock the model for testing."""
        with patch("api.routes.get_model") as mock:
            mock_instance = Mock()

            # Create a test image
            test_image = Image.new("RGB", (512, 512), color="red")
            mock_instance.generate_image.return_value = [test_image]
            mock_instance.image_to_base64.return_value = "test_base64_string"
            mock_instance.health_check.return_value = True
            mock_instance.get_model_info.return_value = {
                "model_name": "test-model",
                "max_image_size": 512,
                "default_steps": 50,
                "max_steps": 100,
                "batch_size": 1,
                "mixed_precision": True,
                "memory_efficient": True,
            }

            mock.return_value = mock_instance
            yield mock_instance

    def test_generate_image_success(self, client, mock_model):
        """Test successful image generation."""
        request_data = {
            "prompt": "a beautiful sunset",
            "num_steps": 30,
            "guidance_scale": 7.5,
            "batch_size": 1
        }

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "images" in data
        assert len(data["images"]) == 1
        assert data["prompt"] == "a beautiful sunset"
        assert "generation_time" in data
        assert data["images"][0]["image_data"] == "test_base64_string"

    def test_generate_image_with_negative_prompt(self, client, mock_model):
        """Test image generation with negative prompt."""
        request_data = {
            "prompt": "a cat",
            "negative_prompt": "dog, blurry",
            "num_steps": 25
        }

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["negative_prompt"] == "dog, blurry"
        mock_model.generate_image.assert_called_once()

    def test_generate_image_with_seed(self, client, mock_model):
        """Test image generation with fixed seed."""
        request_data = {
            "prompt": "a mountain",
            "seed": 12345
        }

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["parameters"]["seed"] == 12345

    def test_generate_image_batch(self, client, mock_model):
        """Test batch image generation."""
        # Mock multiple images
        test_images = [
            Image.new("RGB", (512, 512), color="red"),
            Image.new("RGB", (512, 512), color="blue")
        ]
        mock_model.generate_image.return_value = test_images

        request_data = {
            "prompt": "multiple cats",
            "batch_size": 2
        }

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert len(data["images"]) == 2

    def test_generate_image_invalid_prompt(self, client, mock_model):
        """Test image generation with invalid prompt."""
        request_data = {
            "prompt": "",  # Empty prompt
        }

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_generate_image_invalid_parameters(self, client, mock_model):
        """Test image generation with invalid parameters."""
        request_data = {
            "prompt": "a cat",
            "num_steps": -5,  # Invalid negative steps
            "guidance_scale": 25.0,  # Too high guidance scale
        }

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_generate_image_model_error(self, client, mock_model):
        """Test image generation when model raises error."""
        mock_model.generate_image.side_effect = Exception("Model error")

        request_data = {
            "prompt": "a cat"
        }

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert "error" in data

    @pytest.mark.parametrize("num_steps,expected_steps", [
        (10, 10),  # Valid steps
        (150, 100),  # Should be capped at max_steps
        (None, 50),  # Should use default
    ])
    def test_steps_parameter_validation(self, client, mock_model, num_steps, expected_steps):
        """Test steps parameter validation and capping."""
        request_data = {"prompt": "test"}
        if num_steps is not None:
            request_data["num_steps"] = num_steps

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == 200
        # Verify that the correct steps were passed to the model
        call_args = mock_model.generate_image.call_args
        if num_steps == 150:
            # Should be capped
            assert call_args[0][2] <= 100
        else:
            expected = expected_steps if num_steps is not None else 50
            assert call_args[0][2] == expected


class TestHealthCheckAPI:
    """Test cases for health check API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_check_healthy(self, client):
        """Test health check when service is healthy."""
        with patch("api.routes.get_model") as mock_get_model, \
             patch("api.routes.tf.config.experimental.list_physical_devices") as mock_gpus, \
             patch("api.routes.psutil.virtual_memory") as mock_memory:

            mock_model = Mock()
            mock_model.health_check.return_value = True
            mock_get_model.return_value = mock_model

            mock_gpus.return_value = [Mock()]  # GPU available
            mock_memory.return_value = Mock(total=8000000000, available=4000000000, percent=50.0)

            response = client.get("/api/v1/health")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            assert data["gpu_available"] is True
            assert "memory_usage" in data
            assert "uptime" in data

    def test_health_check_unhealthy(self, client):
        """Test health check when service is unhealthy."""
        with patch("api.routes.get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.health_check.return_value = False
            mock_get_model.return_value = mock_model

            response = client.get("/api/v1/health")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False

    def test_health_check_exception(self, client):
        """Test health check when an exception occurs."""
        with patch("api.routes.get_model") as mock_get_model:
            mock_get_model.side_effect = Exception("Model error")

            response = client.get("/api/v1/health")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False


class TestModelInfoAPI:
    """Test cases for model info API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_get_model_info_success(self, client):
        """Test successful model info retrieval."""
        with patch("api.routes.get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.get_model_info.return_value = {
                "model_name": "test-model",
                "max_image_size": 512,
                "default_steps": 50,
                "max_steps": 100,
                "batch_size": 1,
                "mixed_precision": True,
                "memory_efficient": True,
            }
            mock_get_model.return_value = mock_model

            response = client.get("/api/v1/model/info")

            assert response.status_code == 200
            data = response.json()

            assert data["model_name"] == "test-model"
            assert "model_parameters" in data
            assert "capabilities" in data
            assert data["model_parameters"]["max_image_size"] == 512

    def test_get_model_info_error(self, client):
        """Test model info retrieval when error occurs."""
        with patch("api.routes.get_model") as mock_get_model:
            mock_get_model.side_effect = Exception("Model error")

            response = client.get("/api/v1/model/info")

            assert response.status_code == 500
            data = response.json()
            assert "error" in data


class TestAuthenticationAPI:
    """Test cases for API authentication."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_no_auth_required_by_default(self, client):
        """Test that no authentication is required by default."""
        with patch("api.routes.get_model") as mock_get_model, \
             patch("config.settings.settings.api_key", None):

            mock_model = Mock()
            mock_model.generate_image.return_value = [Image.new("RGB", (512, 512), color="red")]
            mock_model.image_to_base64.return_value = "test_base64"
            mock_get_model.return_value = mock_model

            request_data = {"prompt": "test"}
            response = client.post("/api/v1/generate", json=request_data)

            assert response.status_code == 200

    def test_valid_api_key(self, client):
        """Test access with valid API key."""
        with patch("api.routes.get_model") as mock_get_model, \
             patch("config.settings.settings.api_key", "valid_key"):

            mock_model = Mock()
            mock_model.generate_image.return_value = [Image.new("RGB", (512, 512), color="red")]
            mock_model.image_to_base64.return_value = "test_base64"
            mock_get_model.return_value = mock_model

            headers = {"Authorization": "Bearer valid_key"}
            request_data = {"prompt": "test"}
            response = client.post("/api/v1/generate", json=request_data, headers=headers)

            assert response.status_code == 200

    def test_invalid_api_key(self, client):
        """Test access with invalid API key."""
        with patch("config.settings.settings.api_key", "valid_key"):
            headers = {"Authorization": "Bearer invalid_key"}
            request_data = {"prompt": "test"}
            response = client.post("/api/v1/generate", json=request_data, headers=headers)

            assert response.status_code == 401

    def test_missing_api_key(self, client):
        """Test access without API key when required."""
        with patch("config.settings.settings.api_key", "valid_key"):
            request_data = {"prompt": "test"}
            response = client.post("/api/v1/generate", json=request_data)

            assert response.status_code == 401


class TestRootEndpoints:
    """Test cases for root endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        assert "uptime_seconds" in data
        assert "environment" in data
        assert "model_name" in data