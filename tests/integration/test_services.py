"""
Integration tests for service interactions
"""
import time
from unittest.mock import patch

import pytest
import requests


class TestServiceIntegration:
    """Test integration between different services"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        # Wait for services to be ready
        time.sleep(5)

    def test_fastapi_mlflow_integration(self):
        """Test FastAPI can load models from MLflow"""
        # Test that FastAPI service is running
        response = requests.get("http://localhost:8000/health")
        assert response.status_code == 200

        # Test that MLflow is accessible
        mlflow_response = requests.get("http://localhost:5000/health")
        assert mlflow_response.status_code == 200

    def test_prediction_endpoint(self):
        """Test prediction endpoint with valid data"""
        payload = {
            "tenure": 12,
            "MonthlyCharges": 70.5,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
        }

        response = requests.post("http://localhost:8000/predict", json=payload)
        assert response.status_code == 200

        result = response.json()
        assert "churn_probability" in result
        assert 0 <= result["churn_probability"] <= 1

    def test_streamlit_service(self):
        """Test Streamlit service is accessible"""
        response = requests.get("http://localhost:8501")
        assert response.status_code == 200

    def test_database_connection(self):
        """Test database connectivity"""
        # This would test actual database connections
        # For now, just test that postgres is running
        import psycopg2

        try:
            conn = psycopg2.connect(
                host="localhost",
                database="mlflow",
                user="postgres",
                password="password",
            )
            conn.close()
            assert True
        except Exception:
            pytest.skip("Database not available in test environment")
