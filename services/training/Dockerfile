FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies from PyPI
COPY services/training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/training/churn_mlops_pipeline.py .

# Create data and models directories (will be mounted as volumes)
RUN mkdir -p data models

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

CMD ["python", "churn_mlops_pipeline.py"]
