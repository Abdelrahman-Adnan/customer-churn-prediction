"""
Customer Churn Model Monitoring Script
Monitors data drift and model performance using Evidently and MLflow
Results saved into database to be used with Grafana Dashboard
"""

import datetime
import os
import time

import mlflow
import mlflow.pyfunc
import mlflow.tracking
import numpy as np
import pandas as pd
import psycopg
from evidently import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from prefect import flow, get_run_logger, task

# Configuration
SEND_TIMEOUT = 10
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_REGISTRY_NAME = "customer_churn_classifier"

# Database configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_DB = os.getenv("POSTGRES_DB", "churn_ml_monitoring")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")


def get_db_connection_string():
    """Get database connection string for the environment"""
    return f"host={POSTGRES_HOST} port=5432 dbname={POSTGRES_DB} user={POSTGRES_USER} password={POSTGRES_PASSWORD}"


# Database setup
create_table_statement = """
DROP TABLE IF EXISTS CHURN_ML_METRICS;
CREATE TABLE CHURN_ML_METRICS(
    timestamp TIMESTAMP,
    prediction_drift FLOAT,
    num_drifted_columns INTEGER,
    share_missing_values FLOAT,
    avg_churn_probability FLOAT,
    model_version VARCHAR(10)
);
"""

# Data paths - using your actual dataset
TRAINING_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Feature definitions (same as pipeline and FastAPI)
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_reference_data():
    """Load and prepare reference data from training dataset"""
    # Define required columns (same as pipeline)
    required_columns = numerical_features + categorical_features + ["Churn"]

    # Load training data
    df = pd.read_csv(TRAINING_DATA_PATH, usecols=required_columns)

    # Clean data (same as pipeline)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # Convert target
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

    return df


def load_mlflow_model():
    """Load the latest model from MLflow registry"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        # Get the latest version of the model
        model_version = client.get_latest_versions(
            MODEL_REGISTRY_NAME, stages=["Production", "Staging", "None"]
        )[0]

        # Load the model
        model_uri = f"models:/{MODEL_REGISTRY_NAME}/{model_version.version}"
        model = mlflow.pyfunc.load_model(model_uri)

        return model, model_version.version
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def generate_monitoring_data(reference_data, n_samples=100):
    """Generate synthetic current data for monitoring (simulates new customers)"""
    # Take a random sample from reference data and add some noise
    current_data = reference_data.sample(n=n_samples, random_state=42).copy()

    # Add some drift by modifying numerical features slightly
    for col in numerical_features:
        if col in current_data.columns:
            noise = np.random.normal(0, 0.1, size=len(current_data))
            current_data[col] = current_data[col] * (1 + noise)
            # Ensure values stay positive
            current_data[col] = np.maximum(current_data[col], 0)

    return current_data


def make_predictions(model, data):
    """Make predictions using the MLflow model"""
    if model is None:
        return np.random.random(len(data))  # Fallback random predictions

    try:
        # Prepare data in the format expected by MLflow model
        input_data = data[numerical_features + categorical_features].copy()

        # Make predictions
        predictions = model.predict(input_data)

        # Extract probabilities if available
        if hasattr(predictions, "shape") and len(predictions.shape) > 1:
            return predictions[:, 1]  # Probability of churn
        else:
            return predictions

    except Exception as e:
        print(f"Error making predictions: {e}")
        return np.random.random(len(data))  # Fallback


@task
def prep_db():
    """Prepare database for storing monitoring metrics"""
    logger = get_run_logger()
    logger.info("Preparing database...")

    with psycopg.connect(get_db_connection_string(), autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='ml_metrics_db'")
        if len(res.fetchall()) == 0:
            conn.execute("create database ml_metrics_db;")

        with psycopg.connect(get_db_connection_string()) as conn:
            conn.execute(create_table_statement)


@task
def calculate_metrics_postgresql(curr, reference_data, model, model_version, i):
    """Calculate monitoring metrics and store in PostgreSQL"""
    logger = get_run_logger()

    # Generate current data for this iteration
    current_data = generate_monitoring_data(reference_data, n_samples=50)

    # Make predictions on current data
    predictions = make_predictions(model, current_data)
    current_data["churn_prediction"] = predictions

    # Create and run Evidently report (updated for newer API)
    report = Report(
        metrics=[
            ColumnDriftMetric(column_name="churn_prediction"),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ]
    )

    # Run report without column mapping for simplicity
    report.run(reference_data=reference_data, current_data=current_data)

    # Extract metrics from report
    result = report.as_dict()

    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing_values = result["metrics"][2]["result"]["current"][
        "share_of_missing_values"
    ]
    avg_churn_probability = np.mean(predictions)

    # Insert metrics into database
    curr.execute(
        """INSERT INTO CHURN_ML_METRICS(
            timestamp, prediction_drift, num_drifted_columns,
            share_missing_values, avg_churn_probability, model_version
        ) VALUES (%s, %s, %s, %s, %s, %s)""",
        (
            datetime.datetime.now(),
            prediction_drift,
            num_drifted_columns,
            share_missing_values,
            avg_churn_probability,
            model_version,
        ),
    )

    logger.info(f"Metrics calculated for iteration {i}:")
    logger.info(f"  - Prediction drift: {prediction_drift:.4f}")
    logger.info(f"  - Drifted columns: {num_drifted_columns}")
    logger.info(f"  - Missing values: {share_missing_values:.4f}")
    logger.info(f"  - Avg churn prob: {avg_churn_probability:.4f}")


@flow
def monitoring_main_flow():
    """Main monitoring flow"""
    logger = get_run_logger()

    # Setup
    prep_db()

    # Load reference data and model
    logger.info("Loading reference data and MLflow model...")
    reference_data = load_reference_data()
    model, model_version = load_mlflow_model()

    if model is None:
        logger.warning("Could not load MLflow model, using fallback predictions")
        model_version = "fallback"
    else:
        logger.info(f"Loaded model version: {model_version}")

    logger.info(f"Reference data shape: {reference_data.shape}")

    # Run monitoring iterations
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)

    with psycopg.connect(
        get_db_connection_string(),
        autocommit=True,
    ) as conn:
        for i in range(0, 10):  # Reduced iterations for simplicity
            with conn.cursor() as curr:
                calculate_metrics_postgresql(
                    curr, reference_data, model, model_version, i
                )

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logger.info(f"Sent metrics batch {i+1}/10 to database")


if __name__ == "__main__":
    monitoring_main_flow()
