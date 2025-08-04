# Performance Testing with Locust
from locust import HttpUser, between, task


class ChurnPredictionUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict_single(self):
        """Test single prediction endpoint"""
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
        self.client.post("/predict", json=payload)

    @task(1)
    def health_check(self):
        """Test health endpoint"""
        self.client.get("/health")

    @task(1)
    def get_docs(self):
        """Test API documentation"""
        self.client.get("/docs")
