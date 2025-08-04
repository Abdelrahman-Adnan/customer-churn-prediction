FROM python:3.10-slim

WORKDIR /app

RUN pip install pipenv

COPY docker/training/Pipfile docker/training/Pipfile.lock* ./
RUN pipenv install --system --deploy --ignore-pipfile

COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY config/ ./config/

CMD ["python", "src/churn_mlops_pipeline.py"]
