"""
src/predict.py
Loads the saved model and makes predictions on new article data.
Run from the project root as: python src/predict.py
"""

import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import joblib

import config
from src.preprocessing import load_encoders, apply_encoders


def predict(df):
    """Takes a DataFrame of new articles and returns engagement predictions."""
    model = joblib.load(config.MODEL_PATH)
    topic_encoder, day_encoder = load_encoders()

    df = apply_encoders(df.copy(), topic_encoder, day_encoder)
    X = df[config.FEATURE_COLUMNS]

    return model.predict(X)


if __name__ == "__main__":
    # A few "articles" to test the model on
    mystery_articles = pd.DataFrame([
        {"topic": "Technology", "title_length": 42, "article_length": 1100,
         "publish_hour": 10, "day_of_week": "Tue", "is_trending": 1},
        {"topic": "Sports", "title_length": 30, "article_length": 650,
         "publish_hour": 19, "day_of_week": "Sat", "is_trending": 0},
    ])

    predictions = predict(mystery_articles)
    mystery_articles["predicted_engagement"] = predictions
    print(mystery_articles[["topic", "is_trending", "predicted_engagement"]].to_string(index=False))
