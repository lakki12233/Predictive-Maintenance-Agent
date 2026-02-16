"""
Unit and integration tests for the Oxmaint Predictive Agent API.

Run with: pytest tests/test_api.py -v
"""
import pytest
import base64
from pathlib import Path
from fastapi.testclient import TestClient

# Import the FastAPI app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.main import app


client = TestClient(app)


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def valid_sensor_request():
    """Minimal valid sensor-only request."""
    return {
        "asset_id": "pump_test_001",
        "timestamp": "2026-02-14T10:00:00Z",
        "sensor_window": [
            {"ts": "2026-02-14T10:00:00Z", "sensor_00": 2.5, "sensor_04": 47.0, "sensor_15": 0.8}
        ]
    }


@pytest.fixture
def environmental_critical():
    """Critical environmental conditions."""
    return {
        "operating_hours": 3000,
        "days_since_last_maintenance": 90,
        "ambient_temperature_c": 50,
        "ambient_humidity_percent": 90,
        "load_factor": 1.0,
        "maintenance_overdue": True
    }


@pytest.fixture
def environmental_favorable():
    """Favorable environmental conditions."""
    return {
        "operating_hours": 100,
        "days_since_last_maintenance": 5,
        "ambient_temperature_c": 22,
        "ambient_humidity_percent": 45,
        "load_factor": 0.5,
        "maintenance_overdue": False
    }


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Tests for GET /health endpoint."""
    
    def test_health_returns_ok(self):
        """Health endpoint should return status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
    
    def test_health_includes_model_info(self):
        """Health endpoint should include model configuration."""
        response = client.get("/health")
        data = response.json()
        assert "model_version" in data
        assert "image_model" in data
        assert "paths" in data
        assert "fusion" in data
    
    def test_health_image_model_status(self):
        """Health should indicate if image model is enabled."""
        response = client.get("/health")
        data = response.json()
        assert data["image_model"] in ["enabled", "disabled"]


# =============================================================================
# Single Prediction Tests
# =============================================================================

class TestPredictEndpoint:
    """Tests for POST /predict endpoint."""
    
    def test_predict_sensor_only(self, valid_sensor_request):
        """Should return valid prediction for sensor-only input."""
        response = client.post("/predict", json=valid_sensor_request)
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "asset_id" in data
        assert "failure_probability" in data
        assert "estimated_time_to_breakdown_hours" in data
        assert "top_signals" in data
        assert "inference_ms" in data
        assert "model_version" in data
        assert "explanation" in data
    
    def test_predict_probability_range(self, valid_sensor_request):
        """Failure probability should be between 0 and 1."""
        response = client.post("/predict", json=valid_sensor_request)
        data = response.json()
        assert 0 <= data["failure_probability"] <= 1
    
    def test_predict_ttb_positive(self, valid_sensor_request):
        """Time to breakdown should be non-negative."""
        response = client.post("/predict", json=valid_sensor_request)
        data = response.json()
        assert data["estimated_time_to_breakdown_hours"] >= 0
    
    def test_predict_with_environmental(self, valid_sensor_request, environmental_critical):
        """Should accept environmental data."""
        valid_sensor_request["environmental"] = environmental_critical
        response = client.post("/predict", json=valid_sensor_request)
        assert response.status_code == 200
        data = response.json()
        # Environmental signals should appear
        assert any("env:" in s or "high_" in s for s in data.get("top_signals", []))
    
    def test_predict_environmental_multiplier(self, valid_sensor_request, environmental_critical, environmental_favorable):
        """Critical environment should result in higher risk than favorable."""
        # Get critical risk
        valid_sensor_request["environmental"] = environmental_critical
        response_critical = client.post("/predict", json=valid_sensor_request)
        p_fail_critical = response_critical.json()["failure_probability"]
        
        # Get favorable risk
        valid_sensor_request["environmental"] = environmental_favorable
        response_favorable = client.post("/predict", json=valid_sensor_request)
        p_fail_favorable = response_favorable.json()["failure_probability"]
        
        # Critical should be higher (or equal if both very low)
        assert p_fail_critical >= p_fail_favorable
    
    def test_predict_inference_time_tracked(self, valid_sensor_request):
        """Inference time should be tracked and positive."""
        response = client.post("/predict", json=valid_sensor_request)
        data = response.json()
        assert data["inference_ms"] >= 0
    
    def test_predict_explanation_present(self, valid_sensor_request):
        """Response should include human-readable explanation."""
        response = client.post("/predict", json=valid_sensor_request)
        data = response.json()
        assert data["explanation"] is not None
        assert len(data["explanation"]) > 20  # Should be meaningful text
    
    def test_predict_top_signals_limited(self, valid_sensor_request):
        """Top signals should be limited to 5 items."""
        response = client.post("/predict", json=valid_sensor_request)
        data = response.json()
        assert len(data["top_signals"]) <= 5


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidation:
    """Tests for input validation and error handling."""
    
    def test_missing_asset_id(self):
        """Should reject request without asset_id."""
        request = {
            "timestamp": "2026-02-14T10:00:00Z",
            "sensor_window": [{"ts": "2026-02-14T10:00:00Z", "sensor_00": 1.0}]
        }
        response = client.post("/predict", json=request)
        assert response.status_code == 422
    
    def test_missing_timestamp(self):
        """Should reject request without timestamp."""
        request = {
            "asset_id": "pump_001",
            "sensor_window": [{"ts": "2026-02-14T10:00:00Z", "sensor_00": 1.0}]
        }
        response = client.post("/predict", json=request)
        assert response.status_code == 422
    
    def test_empty_sensor_window(self):
        """Should reject request with empty sensor_window."""
        request = {
            "asset_id": "pump_001",
            "timestamp": "2026-02-14T10:00:00Z",
            "sensor_window": []
        }
        response = client.post("/predict", json=request)
        assert response.status_code == 422
    
    def test_invalid_json(self):
        """Should reject malformed JSON."""
        response = client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_wrong_content_type(self):
        """Should reject non-JSON content type."""
        response = client.post(
            "/predict",
            content="asset_id=pump_001",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == 422


# =============================================================================
# Batch Prediction Tests
# =============================================================================

class TestBatchEndpoint:
    """Tests for POST /predict/batch endpoint."""
    
    def test_batch_single_item(self, valid_sensor_request):
        """Should handle batch with single item."""
        response = client.post("/predict/batch", json=[valid_sensor_request])
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["results"]) == 1
    
    def test_batch_multiple_items(self, valid_sensor_request):
        """Should handle batch with multiple items."""
        batch = [
            {**valid_sensor_request, "asset_id": f"pump_{i:03d}"}
            for i in range(5)
        ]
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 5
        assert len(data["results"]) == 5
    
    def test_batch_inference_time_total(self, valid_sensor_request):
        """Batch should report total inference time."""
        batch = [valid_sensor_request, valid_sensor_request]
        response = client.post("/predict/batch", json=batch)
        data = response.json()
        assert "inference_ms_total" in data
        assert data["inference_ms_total"] >= 0
    
    def test_batch_empty_list(self):
        """Should handle empty batch gracefully."""
        response = client.post("/predict/batch", json=[])
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0


# =============================================================================
# Feature Extraction Tests
# =============================================================================

class TestFeatureExtraction:
    """Tests for sensor feature extraction."""
    
    def test_handles_missing_sensors(self, valid_sensor_request):
        """Should handle requests with missing sensor columns gracefully."""
        # Only provide a few sensors (system should impute or handle missing)
        valid_sensor_request["sensor_window"] = [
            {"ts": "2026-02-14T10:00:00Z", "sensor_00": 1.0}
        ]
        response = client.post("/predict", json=valid_sensor_request)
        # Should not crash - may return 200 or 422 depending on implementation
        assert response.status_code in [200, 422]
    
    def test_handles_multiple_samples(self, valid_sensor_request):
        """Should handle multiple sensor samples for windowed features."""
        valid_sensor_request["sensor_window"] = [
            {"ts": f"2026-02-14T10:{i:02d}:00Z", "sensor_00": float(i), "sensor_04": 45.0 + i}
            for i in range(30)
        ]
        response = client.post("/predict", json=valid_sensor_request)
        assert response.status_code == 200


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
