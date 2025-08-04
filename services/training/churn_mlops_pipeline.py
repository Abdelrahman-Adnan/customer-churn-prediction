"""
Customer Churn Prediction MLOps Pipeline
========================================

This pipeline predicts which customers are likely to cancel their service.

What this pipeline does:
1. Loads and validates customer data
2. Cleans and prepares the data for machine learning
3. Trains multiple AI models with automatic tuning
4. Finds the best performing model
5. Saves the model with detailed tracking
6. Provides comprehensive results and metrics

Features included:
- Automatic data quality checks
- Smart hyperparameter optimization
- Class balancing for better predictions
- Cross-validation for reliable results
- MLflow integration for experiment tracking
- Model registry for production deployment
"""

import warnings

from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

import json
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")
import shutil

import boto3
import cloudpickle
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient

# Prefect
from prefect import flow, get_run_logger, task
from prefect.blocks.system import Secret
from prefect.task_runners import ConcurrentTaskRunner
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ML Libraries
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC

# Configuration - Easy to modify these settings
MLFLOW_TRACKING_URI = "http://34.238.247.213:5000"  # Where MLflow server runs
EXPERIMENT_NAME = "churn_prediction_production"  # Name for this experiment
MODEL_REGISTRY_NAME = "customer_churn_classifier"  # Name for saved models


# Define feature types globally for use in the PyFunc model
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


class ChurnPredictionModel(PythonModel):
    """Custom PyFunc wrapper for our churn prediction model"""

    def load_context(self, context):
        """Load artifacts from MLflow context"""
        import mlflow.sklearn

        self.model = mlflow.sklearn.load_model(context.artifacts["model_path"])
        with open(context.artifacts["encoder_path"], "rb") as f:
            self.encoder = pickle.load(f)
        with open(context.artifacts["scaler_path"], "rb") as f:
            self.scaler = pickle.load(f)

    def predict(self, context, model_input):
        """Preprocess input and make predictions"""
        # Convert to DataFrame
        df = pd.DataFrame(model_input)

        # Preprocess categorical features
        encoded_features = self.encoder.transform(df[categorical_features])
        feature_names = self.encoder.get_feature_names_out(categorical_features)
        X_encoded = pd.DataFrame(encoded_features, columns=feature_names)

        # Combine with numerical features
        X_combined = pd.concat(
            [X_encoded, df[numerical_features].reset_index(drop=True)], axis=1
        )

        # Scale features
        X_scaled = self.scaler.transform(X_combined)

        # Make prediction
        return self.model.predict_proba(X_scaled)[:, 1]


@task(name="setup_mlflow", description="Initialize MLflow tracking")
def setup_mlflow() -> str:
    """
    Set up MLflow to track our experiments

    MLflow helps us:
    - Track model performance
    - Compare different experiments
    - Save models for production use

    Returns:
        experiment_id: ID of the MLflow experiment
    """
    logger = get_run_logger()

    try:
        # Connect to MLflow server
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Test connection
        client = MlflowClient()
        client.search_experiments()

        # Create or get existing experiment
        try:
            experiment = mlflow.create_experiment(EXPERIMENT_NAME)
            experiment_id = experiment
            logger.info(f"Created new MLflow experiment: {EXPERIMENT_NAME}")
        except mlflow.exceptions.MlflowException:
            # Experiment already exists, get its ID
            experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {EXPERIMENT_NAME}")

        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"MLflow experiment ready (ID: {experiment_id})")

        return experiment_id

    except Exception as e:
        logger.warning(f"MLflow server not available: {e}")
        logger.info("Continuing without MLflow tracking...")
        return "local_experiment"


@task(
    name="load_and_validate_data", description="Load data with comprehensive validation"
)
def load_and_validate_data(
    data_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
) -> pd.DataFrame:
    logger = get_run_logger()

    # Define which columns we need for our prediction
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

    try:
        # Load only the columns we need
        required_columns = numerical_features + categorical_features + ["Churn"]
        df = pd.read_csv(data_path, usecols=required_columns)
        logger.info(
            f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns"
        )

        # Start data quality validation
        validation_results = {}

        # Check 1: Make sure all required columns are present
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check 2: Fix data types (TotalCharges is sometimes stored as text)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Check 3: Analyze missing values
        missing_counts = df.isnull().sum()
        missing_percentage = (missing_counts / len(df)) * 100
        validation_results["missing_values"] = missing_counts.to_dict()
        validation_results["missing_percentage"] = missing_percentage.to_dict()

        # Check 4: Warn if too much data is missing
        if missing_percentage.max() > 5:  # More than 5% missing in any column
            high_missing = missing_percentage[missing_percentage > 5]
            logger.warning(f"High missing values detected: {high_missing.to_dict()}")

        # Check 5: Remove rows with missing values
        initial_size = len(df)
        df = df.dropna().reset_index(drop=True)
        final_size = len(df)
        rows_removed = initial_size - final_size
        data_loss_percentage = (rows_removed / initial_size) * 100

        validation_results["rows_removed"] = rows_removed
        validation_results["data_loss_percentage"] = data_loss_percentage

        # Check 6: Analyze target variable distribution
        target_distribution = df["Churn"].value_counts(normalize=True)
        validation_results["target_distribution"] = target_distribution.to_dict()

        # Check 7: Get basic statistics for numerical features
        validation_results["numerical_stats"] = (
            df[numerical_features].describe().to_dict()
        )

        # Check 8: Detect outliers in numerical features
        for col in numerical_features:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (
                (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
            ).sum()
            validation_results[f"{col}_outliers"] = outliers

        # Log validation results
        logger.info(f"Data validation completed:")
        logger.info(f"   • Final dataset shape: {df.shape}")
        logger.info(f"   • Rows removed due to missing values: {rows_removed}")
        logger.info(f"   • Data loss: {data_loss_percentage:.2f}%")
        logger.info(
            f"   • Target distribution: {validation_results['target_distribution']}"
        )

        # Save validation results for later review
        validation_path = Path("data/interim/validation_results.json")
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        with open(validation_path, "w") as f:
            json.dump(validation_results, f, indent=2, default=str)

        logger.info(f"Validation results saved to: {validation_path}")

        return df

    except Exception as e:
        logger.error(f"Data loading/validation failed: {str(e)}")
        raise


@task(name="prepare_data_splits", description="Create train/validation/test splits")
def prepare_data_splits(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger = get_run_logger()

    # First split: separate training data (60%) from the rest (40%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,  # 40% for validation + test
        random_state=42,  # For reproducible results
        stratify=df["Churn"],  # Keep same churn ratio in both sets
    )

    # Second split: divide the remaining 40% into validation (20%) and test (20%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,  # Split the 40% in half
        random_state=42,  # For reproducible results
        stratify=temp_df["Churn"],  # Keep same churn ratio
    )

    # Save the splits for later use or inspection
    splits_dir = Path("data/processed")
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "validation.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    # Log the results
    total_samples = len(df)
    logger.info(f"Data splits created:")
    logger.info(
        f"   • Training:   {train_df.shape[0]:,} samples ({train_df.shape[0]/total_samples*100:.1f}%)"
    )
    logger.info(
        f"   • Validation: {val_df.shape[0]:,} samples ({val_df.shape[0]/total_samples*100:.1f}%)"
    )
    logger.info(
        f"   • Test:       {test_df.shape[0]:,} samples ({test_df.shape[0]/total_samples*100:.1f}%)"
    )
    logger.info(f"Split files saved to: {splits_dir}")

    return train_df, val_df, test_df


@task(name="feature_engineering", description="Comprehensive feature engineering")
def feature_engineering(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger = get_run_logger()

    # Define feature types
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

    def prepare_dataset(df, encoder=None, scaler=None, is_training=True):
        df_processed = df.copy()

        # Convert target variable: "Yes" -> 1, "No" -> 0
        df_processed["Churn"] = df_processed["Churn"].replace({"Yes": 1, "No": 0})

        # One-hot encode categorical features
        # This creates separate columns for each category value
        if is_training:
            encoder = OneHotEncoder(drop="first", sparse_output=False)
            encoded_features = encoder.fit_transform(df_processed[categorical_features])
        else:
            # Use the same encoder fitted on training data
            encoded_features = encoder.transform(df_processed[categorical_features])

        # Create feature names for encoded columns
        feature_names = encoder.get_feature_names_out(categorical_features)
        X_encoded = pd.DataFrame(encoded_features, columns=feature_names)

        # Combine encoded categorical features with numerical features
        X_combined = pd.concat(
            [X_encoded, df_processed[numerical_features].reset_index(drop=True)], axis=1
        )

        # Scale all features to 0-1 range
        # This helps models treat all features equally
        if is_training:
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaled = scaler.fit_transform(X_combined)
        else:
            # Use the same scaler fitted on training data
            X_scaled = scaler.transform(X_combined)

        # Convert back to DataFrame for easier handling
        X_final = pd.DataFrame(X_scaled, columns=X_combined.columns)
        y = df_processed["Churn"].values

        return X_final, y, encoder, scaler

    # Process training data first (this creates the encoder and scaler)
    X_train, y_train, encoder, scaler = prepare_dataset(train_df, is_training=True)

    # Process validation and test data using the same transformations
    X_val, y_val, _, _ = prepare_dataset(val_df, encoder, scaler, is_training=False)
    X_test, y_test, _, _ = prepare_dataset(test_df, encoder, scaler, is_training=False)

    # Save the preprocessing objects for later use in production
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    with open(models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Log the results
    logger.info(f"Feature engineering completed:")
    logger.info(f"   • Total features created: {X_train.shape[1]}")
    logger.info(f"   • Training samples: {X_train.shape[0]:,}")
    logger.info(f"   • Validation samples: {X_val.shape[0]:,}")
    logger.info(f"   • Test samples: {X_test.shape[0]:,}")
    logger.info(f"Preprocessing objects saved to: {models_dir}")

    return X_train, X_val, X_test, y_train, y_val, y_test


@task(name="apply_smote", description="Apply SMOTE for class balancing")
def apply_smote(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance the dataset using SMOTE (Synthetic Minority Oversampling Technique)

    Why we need this:
    - Our dataset likely has more non-churned customers than churned customers
    - This imbalance can make models biased toward predicting "no churn"
    - SMOTE creates synthetic examples of the minority class (churned customers)
    - This helps the model learn patterns for both types of customers equally

    What SMOTE does:
    - Finds similar churned customers in the feature space
    - Creates new synthetic customers between them
    - Results in a balanced dataset with equal numbers of each class

    Args:
        X_train: Training features
        y_train: Training targets (0=no churn, 1=churn)

    Returns:
        X_train_balanced, y_train_balanced: Balanced training data
    """
    logger = get_run_logger()

    # Check the original class distribution
    original_distribution = np.bincount(y_train)
    logger.info(f"Original class distribution:")
    logger.info(f"   • No churn (0): {original_distribution[0]:,} customers")
    logger.info(f"   • Churn (1):    {original_distribution[1]:,} customers")
    logger.info(
        f"   • Imbalance ratio: {original_distribution[0]/original_distribution[1]:.1f}:1"
    )

    # Apply SMOTE to balance the classes
    smote = SMOTE(
        sampling_strategy=1,  # Make classes equal (1:1 ratio)
        random_state=42,  # For reproducible results
    )
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Check the new class distribution
    balanced_distribution = np.bincount(y_train_balanced)
    logger.info(f"Balanced class distribution:")
    logger.info(f"   • No churn (0): {balanced_distribution[0]:,} customers")
    logger.info(f"   • Churn (1):    {balanced_distribution[1]:,} customers")
    logger.info(
        f"   • New ratio: {balanced_distribution[0]/balanced_distribution[1]:.1f}:1"
    )
    logger.info(f"Dataset is now perfectly balanced for better model training")

    feature_names = X_train.columns.tolist()
    X_train_balanced_df = pd.DataFrame(X_train_balanced, columns=feature_names)
    return X_train_balanced_df, pd.Series(y_train_balanced)


@task(
    name="hyperparameter_optimization",
    description="Optimize hyperparameters for a model",
)
def hyperparameter_optimization(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_evals: int = 30,
) -> Dict[str, Any]:
    """Perform hyperparameter optimization for a specific model"""
    logger = get_run_logger()

    def get_model_space(model_name):
        """Define hyperparameter spaces for different models"""
        spaces = {
            "RandomForest": {
                "n_estimators": scope.int(hp.quniform("n_estimators", 100, 500, 50)),
                "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),
                "min_samples_split": scope.int(
                    hp.quniform("min_samples_split", 2, 20, 1)
                ),
                "min_samples_leaf": scope.int(
                    hp.quniform("min_samples_leaf", 1, 10, 1)
                ),
                "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
                "bootstrap": hp.choice("bootstrap", [True, False]),
            },
            "XGBoost": {
                "n_estimators": scope.int(hp.quniform("n_estimators", 50, 200, 1)),
                "max_depth": scope.int(hp.quniform("max_depth", 3, 10, 1)),
                "learning_rate": hp.loguniform(
                    "learning_rate", np.log(0.01), np.log(0.3)
                ),
                "min_child_weight": hp.uniform("min_child_weight", 1, 10),
                "subsample": hp.uniform("subsample", 0.5, 1.0),
                "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
            },
            "LightGBM": {
                "n_estimators": scope.int(hp.quniform("n_estimators", 50, 200, 1)),
                "max_depth": scope.int(hp.quniform("max_depth", 3, 10, 1)),
                "learning_rate": hp.loguniform(
                    "learning_rate", np.log(0.01), np.log(0.3)
                ),
                "num_leaves": scope.int(hp.quniform("num_leaves", 20, 100, 1)),
                "min_child_weight": hp.uniform("min_child_weight", 1, 10),
                "subsample": hp.uniform("subsample", 0.5, 1.0),
                "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
            },
        }
        return spaces.get(model_name, {})

    def get_model_instance(model_name, params):
        """Get model instance with hyperparameters"""
        # Convert hyperopt choices to proper types
        if model_name == "RandomForest":
            # Map choice indices to actual values
            max_features_options = ["sqrt", "log2", None]
            bootstrap_options = [True, False]
            if "max_features" in params:
                params["max_features"] = max_features_options[
                    int(params["max_features"])
                ]
            if "bootstrap" in params:
                params["bootstrap"] = bootstrap_options[int(params["bootstrap"])]

            # Convert numeric parameters to int
            int_params = [
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
            ]
            for p in int_params:
                if p in params:
                    params[p] = int(params[p])

            return RandomForestClassifier(**params, random_state=42)

        elif model_name == "XGBoost":
            # Convert numeric parameters to int
            params["n_estimators"] = int(params["n_estimators"])
            params["max_depth"] = int(params["max_depth"])
            return xgb.XGBClassifier(**params, random_state=42, eval_metric="logloss")

        elif model_name == "LightGBM":
            # Convert numeric parameters to int
            params["n_estimators"] = int(params["n_estimators"])
            params["max_depth"] = int(params["max_depth"])
            params["num_leaves"] = int(params["num_leaves"])
            return lgb.LGBMClassifier(**params, random_state=42, verbose=-1)

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def objective(params):
        """Objective function for hyperparameter optimization"""
        try:
            # Create model with current parameters
            model = get_model_instance(model_name, params.copy())

            # Train and evaluate
            model.fit(X_train.values, y_train.values)
            y_pred = model.predict(X_val.values)

            # Calculate recall score (what we want to maximize)
            recall = recall_score(y_val, y_pred)

            # Hyperopt minimizes, so return negative recall
            return {"loss": -recall, "status": STATUS_OK}

        except Exception as e:
            logger.warning(f"Error in optimization trial: {e}")
            return {"loss": 0, "status": STATUS_OK}

    # Get hyperparameter space for this model
    space = get_model_space(model_name)

    if not space:
        logger.warning(f"No hyperparameter space defined for {model_name}")
        return {"best_params": {}, "best_score": 0, "trials": None}

    logger.info(f"Starting hyperparameter optimization for {model_name}...")

    # Initialize trials object to track results
    trials = Trials()

    try:
        # Run Bayesian optimization
        best_params = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            verbose=False,
        )

        # Get the best score
        best_loss = min([trial["result"]["loss"] for trial in trials.trials])
        best_score = -best_loss

        logger.info(f"Optimization completed for {model_name}")
        logger.info(f"Best recall score: {best_score:.4f}")

        return {"best_params": best_params, "best_score": best_score, "trials": trials}

    except Exception as e:
        logger.error(f"Optimization failed for {model_name}: {e}")
        return {"best_params": {}, "best_score": 0, "trials": None}


@task(name="train_baseline_models", description="Train baseline models")
def train_baseline_models(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    logger = get_run_logger()

    # Define baseline models with default parameters
    baseline_models = {
        "SVC": SVC(
            kernel="linear",  # Linear decision boundary
            probability=True,  # Enable probability predictions
            random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(
            random_state=42, max_iter=1000  # Enough iterations to converge
        ),
    }

    results = {}

    # Train and evaluate each baseline model
    for model_name, model in baseline_models.items():
        logger.info(f"Training baseline model: {model_name}")

        # Track this experiment in MLflow
        with mlflow.start_run(run_name=f"{model_name}_baseline"):
            # Train the model
            model.fit(X_train.values, y_train.values)

            # Make predictions on validation set
            y_pred = model.predict(X_val.values)
            y_pred_proba = (
                model.predict_proba(X_val)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

            # Calculate performance metrics
            metrics = {
                "accuracy": accuracy_score(y_val, y_pred),  # Overall correctness
                "precision": precision_score(
                    y_val, y_pred
                ),  # True positives / (True + False positives)
                "recall": recall_score(
                    y_val, y_pred
                ),  # True positives / (True + False negatives)
                "f1": f1_score(y_val, y_pred),  # Harmonic mean of precision and recall
            }

            # Add ROC-AUC if model supports probability predictions
            if y_pred_proba is not None:
                metrics["roc_auc"] = roc_auc_score(y_val, y_pred_proba)

            # Log everything to MLflow for tracking
            mlflow.log_params(model.get_params())
            mlflow.log_param("model_type", f"{model_name}_baseline")
            mlflow.log_metrics(metrics)

            # Store results for comparison
            results[model_name] = {
                "model": model,
                "metrics": metrics,
                "predictions": y_pred,
                "probabilities": y_pred_proba,
            }

            # Log performance
            logger.info(f"   {model_name} performance:")
            logger.info(f"      • Accuracy:  {metrics['accuracy']:.3f}")
            logger.info(f"      • Precision: {metrics['precision']:.3f}")
            logger.info(f"      • Recall:    {metrics['recall']:.3f}")
            logger.info(f"      • F1-Score:  {metrics['f1']:.3f}")

    logger.info("Baseline model training completed")
    return results


@task(
    name="train_optimized_models",
    description="Train models with optimized hyperparameters",
)
def train_optimized_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    optimization_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Train models with optimized hyperparameters"""
    logger = get_run_logger()
    results = {}

    for model_name, opt_result in optimization_results.items():
        logger.info(f"Training optimized model: {model_name}")

        # Check if optimization was successful
        if not opt_result or "best_params" not in opt_result:
            logger.warning(f"No optimization results for {model_name}, skipping...")
            continue

        best_params = opt_result["best_params"].copy()

        # Skip if no parameters were found
        if not best_params:
            logger.warning(f"No best parameters found for {model_name}, skipping...")
            continue

        try:
            with mlflow.start_run(run_name=f"{model_name}_optimized"):
                # Create model with best parameters
                if model_name == "RandomForest":
                    # Handle choice parameters correctly
                    max_features_options = ["sqrt", "log2", None]
                    bootstrap_options = [True, False]

                    if "max_features" in best_params:
                        best_params["max_features"] = max_features_options[
                            int(best_params["max_features"])
                        ]
                    if "bootstrap" in best_params:
                        best_params["bootstrap"] = bootstrap_options[
                            int(best_params["bootstrap"])
                        ]

                    # Convert numeric parameters to int
                    int_params = [
                        "n_estimators",
                        "max_depth",
                        "min_samples_split",
                        "min_samples_leaf",
                    ]
                    for p in int_params:
                        if p in best_params:
                            best_params[p] = int(best_params[p])

                    model = RandomForestClassifier(**best_params, random_state=42)

                elif model_name == "XGBoost":
                    # Convert numeric parameters to int
                    for param in ["n_estimators", "max_depth"]:
                        if param in best_params:
                            best_params[param] = int(best_params[param])

                    model = xgb.XGBClassifier(
                        **best_params,
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric="logloss",
                    )

                elif model_name == "LightGBM":
                    # Convert numeric parameters to int
                    for param in ["n_estimators", "max_depth", "num_leaves"]:
                        if param in best_params:
                            best_params[param] = int(best_params[param])

                    model = lgb.LGBMClassifier(
                        **best_params, random_state=42, verbose=-1
                    )

                else:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue

                # Train model
                model.fit(X_train.values, y_train.values)

                # Make predictions
                y_pred = model.predict(X_val.values)
                y_pred_proba = model.predict_proba(X_val)[:, 1]

                # Calculate metrics
                metrics = {
                    "accuracy": accuracy_score(y_val, y_pred),
                    "precision": precision_score(y_val, y_pred),
                    "recall": recall_score(y_val, y_pred),
                    "f1": f1_score(y_val, y_pred),
                    "roc_auc": roc_auc_score(y_val, y_pred_proba),
                }

                # Log to MLflow
                mlflow.log_params(best_params)
                mlflow.log_param("model_type", f"{model_name}_optimized")
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, f"{model_name}_model")

                # Store results
                results[model_name] = {
                    "model": model,
                    "metrics": metrics,
                    "best_params": best_params,
                    "predictions": y_pred,
                    "probabilities": y_pred_proba,
                }

                # Log performance
                logger.info(f"   {model_name} optimized performance:")
                logger.info(f"      • Accuracy:  {metrics['accuracy']:.3f}")
                logger.info(f"      • Precision: {metrics['precision']:.3f}")
                logger.info(f"      • Recall:    {metrics['recall']:.3f}")
                logger.info(f"      • F1-Score:  {metrics['f1']:.3f}")
                logger.info(f"      • ROC-AUC:   {metrics['roc_auc']:.3f}")

        except Exception as e:
            logger.error(f"Failed to train optimized {model_name}: {e}")
            continue

    logger.info(
        f"Optimized model training completed. Successfully trained {len(results)} models."
    )
    return results


@task(
    name="cross_validate_models", description="Perform cross-validation on all models"
)
def cross_validate_models(
    all_models: Dict[str, Dict[str, Any]], X_train: np.ndarray, y_train: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Perform cross-validation on all trained models"""
    logger = get_run_logger()

    cv_results = {}
    scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    logger.info("Starting cross-validation for all models...")

    for model_name, model_data in all_models.items():
        logger.info(f"Cross-validating: {model_name}")

        model = model_data["model"]

        # Perform cross-validation
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring=scoring_metrics,
            return_train_score=True,
            n_jobs=-1,
        )

        # Calculate statistics
        results = {}
        for metric in scoring_metrics:
            test_scores = cv_scores[f"test_{metric}"]
            results[f"{metric}_mean"] = test_scores.mean()
            results[f"{metric}_std"] = test_scores.std()

        cv_results[model_name] = results

        # Log cross-validation results to MLflow
        with mlflow.start_run(run_name=f"{model_name}_cv"):
            mlflow.log_param("model_type", f"{model_name}_cv")
            mlflow.log_metrics(results)

    logger.info("Cross-validation completed for all models")
    return cv_results


@task(name="select_best_model", description="Select the best performing model")
def select_best_model(
    all_models: Dict[str, Dict[str, Any]],
    cv_results: Dict[str, Dict[str, float]],
    selection_metric: str = "recall_mean",
) -> Dict[str, Any]:
    """Select the best model based on cross-validation results"""
    logger = get_run_logger()

    # Find best model based on selection metric
    best_score = -1
    best_model_name = None

    for model_name, cv_metrics in cv_results.items():
        score = cv_metrics.get(selection_metric, 0)
        if score > best_score:
            best_score = score
            best_model_name = model_name

    if best_model_name is None:
        raise ValueError("No best model found")

    best_model_data = all_models[best_model_name]
    best_cv_results = cv_results[best_model_name]

    logger.info(f"Best model selected: {best_model_name}")
    logger.info(f"Best {selection_metric}: {best_score:.4f}")

    # Prepare model info for registry
    model_info = {
        "model_name": best_model_name,
        "model": best_model_data["model"],
        "metrics": best_model_data["metrics"],
        "cv_metrics": best_cv_results,
        "best_score": best_score,
        "selection_metric": selection_metric,
    }

    return model_info


@task(
    name="register_best_model",
    description="Register best model to MLflow Model Registry",
)
def register_best_model(
    model_info: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
) -> str:
    """Register the best model to MLflow Model Registry"""
    logger = get_run_logger()

    model_name = model_info["model_name"]
    model = model_info["model"]

    with mlflow.start_run(run_name=f"{model_name}_final"):
        # 1. Final evaluation on test set
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        test_metrics = {
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred),
            "test_recall": recall_score(y_test, y_test_pred),
            "test_f1": f1_score(y_test, y_test_pred),
            "test_roc_auc": roc_auc_score(y_test, y_test_proba),
        }

        # 2. Log metrics and parameters
        mlflow.log_metrics(test_metrics)
        mlflow.log_metrics(model_info["cv_metrics"])
        mlflow.log_param("model_type", f"{model_name}_final")
        mlflow.log_param("selection_metric", model_info["selection_metric"])
        mlflow.log_param("selection_score", model_info["best_score"])

        # 3. Save preprocessing artifacts locally
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save paths for MLflow artifacts
        encoder_path = artifacts_dir / "encoder.pkl"
        scaler_path = artifacts_dir / "scaler.pkl"
        model_path = artifacts_dir / "model"
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        # Load existing preprocessing objects
        with open("models/encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Save to artifacts directory
        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        mlflow.sklearn.save_model(model, model_path)

        # 4. Create custom PyFunc model
        python_model = ChurnPredictionModel()
        artifacts = {
            "encoder_path": str(encoder_path),
            "scaler_path": str(scaler_path),
            "model_path": str(model_path),
        }

        # 5. Log the PyFunc model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=python_model,
            artifacts=artifacts,
            registered_model_name=MODEL_REGISTRY_NAME,
        )

        # 6. Add model version description
        client = MlflowClient()
        model_version = client.get_latest_versions(MODEL_REGISTRY_NAME)[0]

        description = f"""
        Best performing model: {model_name}
        Selection metric: {model_info['selection_metric']} = {model_info['best_score']:.4f}
        Test Performance:
        - Accuracy: {test_metrics['test_accuracy']:.4f}
        - Precision: {test_metrics['test_precision']:.4f}
        - Recall: {test_metrics['test_recall']:.4f}
        - F1-Score: {test_metrics['test_f1']:.4f}
        - ROC-AUC: {test_metrics['test_roc_auc']:.4f}

        This model includes preprocessing pipeline for end-to-end predictions.
        """

        client.update_model_version(
            name=MODEL_REGISTRY_NAME,
            version=model_version.version,
            description=description,
        )

        model_uri = f"models:/{MODEL_REGISTRY_NAME}/{model_version.version}"
        logger.info(f"Model registered: {model_uri}")
        logger.info(f"Test metrics: {test_metrics}")

        return model_uri


@flow(
    name="customer-churn-mlops-pipeline",
    description="Complete MLOps pipeline for customer churn prediction",
    task_runner=ConcurrentTaskRunner(),
)
def churn_mlops_pipeline(
    data_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    hyperopt_max_evals: int = 5,  # Reduced for faster execution
):
    logger = get_run_logger()
    logger.info("Starting Customer Churn MLOps Pipeline...")
    logger.info("=" * 60)

    # Step 1: Set up experiment tracking
    logger.info("Step 1: Setting up MLflow experiment tracking...")
    experiment_id = setup_mlflow()

    # Step 2: Load and validate the raw data
    logger.info("Step 2: Loading and validating customer data...")
    df = load_and_validate_data(data_path)

    # Step 3: Create train/validation/test splits
    logger.info("Step 3: Creating data splits...")
    train_df, val_df, test_df = prepare_data_splits(df)

    # Step 4: Feature engineering (convert data for ML)
    logger.info("Step 4: Engineering features for machine learning...")
    X_train, X_val, X_test, y_train, y_val, y_test = feature_engineering(
        train_df, val_df, test_df
    )

    # Step 5: Balance the training data
    logger.info("Step 5: Balancing training data with SMOTE...")
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # Step 6: Train baseline models
    logger.info("Step 6: Training baseline models...")
    baseline_results = train_baseline_models(
        X_train_balanced, y_train_balanced, X_val, y_val
    )

    # Step 7: Optimize hyperparameters for advanced models
    logger.info("Step 7: Optimizing hyperparameters for advanced models...")
    optimization_models = ["RandomForest", "XGBoost", "LightGBM"]
    optimization_results = {}

    for model_name in optimization_models:
        logger.info(f"   Optimizing {model_name} (max {hyperopt_max_evals} trials)...")
        opt_result = hyperparameter_optimization(
            model_name,
            X_train_balanced,
            y_train_balanced,
            X_val,
            y_val,
            hyperopt_max_evals,
        )
        optimization_results[model_name] = opt_result

    # Step 8: Train optimized models
    logger.info("Step 8: Training optimized models...")
    optimized_results = train_optimized_models(
        X_train_balanced, y_train_balanced, X_val, y_val, optimization_results
    )

    # Step 9: Combine all model results
    logger.info("Step 9: Combining all model results...")
    all_models = {**baseline_results, **optimized_results}
    logger.info(f"   Total models trained: {len(all_models)}")

    # Step 10: Cross-validation for robust evaluation
    logger.info("Step 10: Performing cross-validation...")
    cv_results = cross_validate_models(all_models, X_train_balanced, y_train_balanced)

    # Step 11: Select the best performing model
    logger.info("Step 11: Selecting best model...")
    best_model_info = select_best_model(all_models, cv_results, "recall_mean")

    # Step 12: Register the best model for production
    logger.info("Step 12: Registering best model...")
    model_uri = register_best_model(best_model_info, X_test, y_test)

    # Step 13: Generate final summary
    logger.info("Step 13: Generating pipeline summary...")
    summary = {
        "pipeline_status": "SUCCESS",
        "best_model": best_model_info["model_name"],
        "best_score": best_model_info["best_score"],
        "selection_metric": best_model_info["selection_metric"],
        "model_uri": model_uri,
        "total_models_trained": len(all_models),
        "experiment_id": experiment_id,
    }

    # Final success message
    logger.info("=" * 60)
    logger.info("MLOps Pipeline completed successfully!")
    logger.info(f"Best model: {summary['best_model']}")
    logger.info(f"Best {summary['selection_metric']}: {summary['best_score']:.4f}")
    logger.info(f"Model URI: {summary['model_uri']}")
    logger.info(f"MLflow experiment ID: {summary['experiment_id']}")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    print("Starting Customer Churn Prediction MLOps Pipeline")
    print("=" * 60)
    print("This will train multiple AI models to predict customer churn")
    print("and automatically select the best performing one.")
    print("=" * 60)

    # Run the complete pipeline
    result = churn_mlops_pipeline()

    # Display final results
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Best Model: {result['best_model']}")
    print(f"Performance Score: {result['best_score']:.4f}")
    print(f"Total Models Trained: {result['total_models_trained']}")
    print(f"Model URI: {result['model_uri']}")
