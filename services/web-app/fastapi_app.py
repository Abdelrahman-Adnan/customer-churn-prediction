import logging
from typing import Any, Dict, Optional

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

# Configuration
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_REGISTRY_NAME = "customer_churn_classifier"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using MLflow registered models",
    version="2.0.0",
)

model = None
model_name = None
model_version = None


# Input data model
class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    Partner: str
    Dependents: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


@app.on_event("startup")
async def load_model():
    """Load the latest model from MLflow registry"""
    global model, model_name, model_version

    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Get latest model version
        client = MlflowClient()
        latest_versions = client.get_latest_versions(MODEL_REGISTRY_NAME)

        if not latest_versions:
            logger.error(f"No models found in registry: {MODEL_REGISTRY_NAME}")
            raise HTTPException(status_code=500, detail="No models available")

        latest_version = latest_versions[0]
        model_version = latest_version.version

        # Load model
        model_uri = f"models:/{MODEL_REGISTRY_NAME}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Extract model name from description or run data
        try:
            # Get run details to extract model name
            run_id = latest_version.run_id
            run = client.get_run(run_id)
            model_name = run.data.params.get("model_type", "Unknown").replace(
                "_final", ""
            )
        except:
            model_name = "MLflow Model"

        logger.info(f"✅ Model loaded: {model_name} (v{model_version})")
        logger.info(f"📍 Model URI: {model_uri}")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")


@app.get("/")
async def root():
    """API homepage with model information"""
    return {
        "message": "Customer Churn Prediction API",
        "status": "healthy" if model else "model_not_loaded",
        "model_name": model_name,
        "model_version": model_version,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model-info",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_name": model_name,
        "model_version": model_version,
    }


@app.get("/model-info")
async def get_model_info():
    """Get detailed model information"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": model_name,
        "model_version": model_version,
        "registry_name": MODEL_REGISTRY_NAME,
        "mlflow_uri": MLFLOW_TRACKING_URI,
    }


@app.post("/predict")
async def predict_churn(customer_data: CustomerData) -> Dict[str, Any]:
    """
    Predict customer churn probability

    Args:
        customer_data: Customer information

    Returns:
        Prediction result with churn probability and model info
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([customer_data.dict()])

        # Make prediction (preprocessing handled by PyFunc model)
        churn_prob = model.predict(input_df)[0]

        return {
            "churn_probability": float(churn_prob),
            "model_name": model_name,
            "model_version": model_version,
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
