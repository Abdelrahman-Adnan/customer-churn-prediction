FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies from PyPI
COPY services/web-app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/web-app/ .

# Create necessary directories
RUN mkdir -p data models

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Start both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 & wait"]
