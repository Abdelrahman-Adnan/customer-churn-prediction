FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies from PyPI
COPY services/monitoring/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/monitoring/monitor_churn_model.py .

# Create data and models directories (will be mounted as volumes)
RUN mkdir -p data models

ENV POSTGRES_HOST=postgres
ENV POSTGRES_DB=churn_ml_monitoring
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

CMD ["python", "monitor_churn_model.py"]
