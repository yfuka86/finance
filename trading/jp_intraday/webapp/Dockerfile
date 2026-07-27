# Web dashboard receiver for trade.a-tokyo.jp (Cloud Run).
# Build context = this directory:  gcloud run deploy --source trading/jp_intraday/webapp
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
