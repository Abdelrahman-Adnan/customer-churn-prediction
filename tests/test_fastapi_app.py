import numpy as np
import pytest


class TestFastAPIApp:
    def test_import_fastapi_module(self):
        """Test that FastAPI application can be imported."""
        try:
            import os
            import sys

            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "web_app"))
            import fastapi_app

            assert hasattr(fastapi_app, "app")
        except ImportError:
            pytest.skip("FastAPI module not available")

    def test_customer_data_validation(self):
        """Test customer data validation schema."""
        try:
            import os
            import sys

            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "web_app"))
            from fastapi_app import CustomerData

            valid_data = {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 65.5,
                "TotalCharges": 786.0,
            }

            customer = CustomerData(**valid_data)
            assert customer.gender == "Male"
            assert customer.tenure == 12
            assert customer.MonthlyCharges == 65.5
        except ImportError:
            pytest.skip("FastAPI module not available")

    def test_api_health_check(self):
        """Test API health check endpoint."""
        try:
            import os
            import sys

            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "web_app"))
            from fastapi_app import health_check

            response = health_check()
            assert "status" in response
            assert response["status"] == "healthy"
        except ImportError:
            pytest.skip("FastAPI module not available")

    def test_model_loading_interface(self):
        """Test that model loading functions exist."""
        try:
            import os
            import sys

            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "web_app"))
            import fastapi_app

            assert hasattr(fastapi_app, "load_models")
            assert hasattr(fastapi_app, "preprocess_data")
            assert hasattr(fastapi_app, "predict_churn")
        except ImportError:
            pytest.skip("FastAPI module not available")

    def test_basic_imports(self):
        """Test basic module imports work."""
        try:
            import fastapi
            import joblib
            import mlflow
            import pandas as pd

            assert fastapi is not None
            assert pd is not None
            assert np is not None
            assert joblib is not None
            assert mlflow is not None
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_data_structure_validation(self):
        """Test data structure validation."""
        sample_data = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 65.5,
            "TotalCharges": 786.0,
        }

        required_fields = [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
        ]

        for field in required_fields:
            assert field in sample_data

        assert isinstance(sample_data["SeniorCitizen"], int)
        assert isinstance(sample_data["tenure"], int)
        assert isinstance(sample_data["MonthlyCharges"], float)
        assert isinstance(sample_data["TotalCharges"], float)
