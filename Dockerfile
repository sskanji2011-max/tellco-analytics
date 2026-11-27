FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
RUN pip install --no-cache-dir ".[dashboard,api]"
# data/processed is produced by:  docker run -v $PWD/data:/app/data tellco tellco-run --input data/raw/telcom_data.xlsx
EXPOSE 8501 8000
# model API:  docker run -p 8000:8000 -v $PWD/models:/app/models tellco uvicorn app.predict_api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
