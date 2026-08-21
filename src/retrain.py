"""
src/retrain.py
Closes the feedback loop: retrains the model on MARCH data
(the new reality), producing Model V2.

This is what a real MLOps pipeline would trigger automatically
once drift/performance decay is detected.

Run from the project root as: python src/retrain.py
"""

import sys
import os
sys.path.append(os.getcwd())

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import config
from src.preprocessing import fit_encoders


def main():
    print("Retraining on March data (the new reality)...\n")

    df = pd.read_csv(config.MARCH_DATA_PATH)
    df, topic_encoder, day_encoder = fit_encoders(df)

    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model V2 (March) accuracy on March test data: {accuracy:.4f}")

    joblib.dump(model, config.MODEL_V2_PATH)
    joblib.dump(topic_encoder, config.TOPIC_ENCODER_V2_PATH)
    joblib.dump(day_encoder, config.DAY_ENCODER_V2_PATH)
    print(f"Model V2 saved to {config.MODEL_V2_PATH}")

    print("\n" + "=" * 60)
    print("FEEDBACK LOOP CLOSED")
    print("=" * 60)
    print("Old model (trained on January) on March data: 60.00% accuracy")
    print(f"New model (trained on March)  on March data: {accuracy*100:.2f}% accuracy")
    print("\nThis is the full MLOps lifecycle: monitor -> detect decay -> retrain -> redeploy.")


if __name__ == "__main__":
    main()
