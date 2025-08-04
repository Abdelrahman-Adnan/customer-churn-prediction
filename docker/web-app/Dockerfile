FROM python:3.10-slim

WORKDIR /app

RUN pip install pipenv

COPY docker/web-app/Pipfile docker/web-app/Pipfile.lock* ./
RUN pipenv install --system --deploy --ignore-pipfile

COPY web_app/ ./web_app/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn web_app.fastapi_app:app --host 0.0.0.0 --port 8000 & streamlit run web_app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 & wait"]
