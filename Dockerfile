# CPU-only image; python:3.10-slim matches the version the model was baked with.
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
# Self-heal: model weights ship in the repo, but retrain (~30s, offline) if absent.
RUN test -f models/sign_model.pkl || python train_model.py
EXPOSE 8501
HEALTHCHECK --interval=5s --timeout=3s --retries=40 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501')" || exit 1
CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0","--browser.gatherUsageStats=false"]
