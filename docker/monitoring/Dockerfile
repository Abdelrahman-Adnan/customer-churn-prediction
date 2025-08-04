FROM python:3.10-slim

WORKDIR /app

RUN pip install pipenv

COPY docker/monitoring/Pipfile docker/monitoring/Pipfile.lock* ./
RUN pipenv install --system --deploy --ignore-pipfile

COPY src/simple_monitor_churn.py ./
COPY data/processed/ ./data/processed/
COPY models/ ./models/

CMD ["python", "simple_monitor_churn.py"]
