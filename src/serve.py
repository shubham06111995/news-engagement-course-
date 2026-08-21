"""
src/serve.py
Serves the model as a live REST API using FastAPI.
Now instrumented with Prometheus metrics - tracks request counts,
latency, and prediction outcomes, so we can monitor it in production.

Run locally with: uvicorn src.serve:app --host 0.0.0.0 --port 8000
"""

import sys
import os
import time
sys.path.append(os.getcwd())

import joblib
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

import config
from src.preprocessing import load_encoders, apply_encoders

app = FastAPI(title="News Article Engagement Predictor")

model = joblib.load(config.MODEL_PATH)
topic_encoder, day_encoder = load_encoders()

# --- Prometheus metrics ---
# Counter: counts things that only go up (total requests, total predictions per class)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total number of predictions made", ["predicted_class"]
)
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests received", ["endpoint"]
)
# Histogram: tracks how long things take (request latency)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Time taken to process a prediction request"
)


class Article(BaseModel):
    topic: str
    title_length: int
    article_length: int
    publish_hour: int
    day_of_week: str
    is_trending: int


@app.get("/health")
def health():
    REQUEST_COUNT.labels(endpoint="/health").inc()
    return {"status": "ok", "version": "v2-deployed-via-cicd"}


@app.get("/metrics")
def metrics():
    """Prometheus scrapes THIS endpoint to collect our metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(article: Article):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="/predict").inc()

    df = pd.DataFrame([article.dict()])
    df = apply_encoders(df, topic_encoder, day_encoder)
    X = df[config.FEATURE_COLUMNS]
    prediction = model.predict(X)[0]

    PREDICTION_COUNT.labels(predicted_class=prediction).inc()
    REQUEST_LATENCY.observe(time.time() - start_time)

    return {"predicted_engagement": prediction}