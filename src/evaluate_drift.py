"""
src/evaluate_drift.py
Tests the January-trained model on March data (WITHOUT retraining)
to demonstrate data drift with real accuracy numbers.
Run from the project root as: python src/evaluate_drift.py
"""

import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

import config
from src.preprocessing import load_encoders, apply_encoders


def main():
    df = pd.read_csv(config.MARCH_DATA_PATH)

    model = joblib.load(config.MODEL_PATH)
    topic_encoder, day_encoder = load_encoders()

    df = apply_encoders(df.copy(), topic_encoder, day_encoder)
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    print(f"Same Model V1 (trained on January) accuracy on MARCH data: {accuracy:.4f}")
    print("\nWhat the model got wrong:")
    df["predicted"] = predictions
    wrong = df[df[config.TARGET_COLUMN] != df["predicted"]]
    print(wrong[["topic", "is_trending", config.TARGET_COLUMN, "predicted"]].to_string(index=False))


if __name__ == "__main__":
    main()
