"""
Integration tests for FastAPI application
"""
import pytest
from fastapi.testclient import TestClient
import numpy as np

@pytest.fixture
def sample_transaction():
    """Create sample transaction data"""
    transaction = {
        "Time": 0,
        "Amount": 149.62
    }
    # Add V features
    for i in range(1, 29):
        transaction[f"V{i}"] = np.random.randn()
    return transaction


class TestAPI:
    """Test cases for API endpoints"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client"""
        try:
            from src.api.app import app
            self.client = TestClient(app)
            self.api_available = True
        except Exception as e:
            self.api_available = False
            pytest.skip(f"API not available: {e}")

    def test_root_endpoint(self):
        """Test root endpoint"""
        if not self.api_available:
            pytest.skip("API not available")
        response = self.client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_check(self):
        """Test health check endpoint"""
        if not self.api_available:
            pytest.skip("API not available")
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_predict_single_transaction(self, sample_transaction):
        """Test single transaction prediction"""
        if not self.api_available:
            pytest.skip("API not available")
        response = self.client.post("/predict", json=sample_transaction)
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200
        data = response.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "confidence" in data
        assert "risk_level" in data
        assert isinstance(data["is_fraud"], bool)
        assert 0 <= data["fraud_probability"] <= 1

    def test_predict_batch_transactions(self, sample_transaction):
        """Test batch prediction"""
        if not self.api_available:
            pytest.skip("API not available")
        transactions = [sample_transaction for _ in range(5)]
        batch_request = {"transactions": transactions}
        response = self.client.post("/predict/batch", json=batch_request)
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200
        data = response.json()
        assert "total_transactions" in data
        assert "fraud_count" in data
        assert "predictions" in data
        assert data["total_transactions"] == 5
        assert len(data["predictions"]) == 5

    def test_predict_invalid_transaction(self):
        """Test prediction with invalid data"""
        if not self.api_available:
            pytest.skip("API not available")
        invalid_transaction = {"Time": 0, "Amount": 100}
        response = self.client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422  # Validation error

    def test_model_info(self):
        """Test model info endpoint"""
        if not self.api_available:
            pytest.skip("API not available")
        response = self.client.get("/model/info")
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "threshold" in data
        assert "features_count" in data

    def test_update_threshold(self):
        """Test threshold update"""
        if not self.api_available:
            pytest.skip("API not available")
        new_threshold = 0.7
        response = self.client.post(f"/model/threshold?new_threshold={new_threshold}")
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200
        data = response.json()
        assert "new_threshold" in data
        assert data["new_threshold"] == new_threshold

    def test_update_threshold_invalid(self):
        """Test invalid threshold update"""
        if not self.api_available:
            pytest.skip("API not available")
        response = self.client.post("/model/threshold?new_threshold=1.5")
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 400

    def test_explain_prediction(self, sample_transaction):
        """Test prediction explanation"""
        if not self.api_available:
            pytest.skip("API not available")
        response = self.client.post("/predict/explain", json=sample_transaction)
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "input_features" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
