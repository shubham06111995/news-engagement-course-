"""
src/train.py
Trains the News Article Engagement model on January data.
Run from the project root as: python src/train.py
"""

import sys
import os
sys.path.append(os.getcwd())  # allows importing config.py from project root

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

import config
from src.preprocessing import fit_encoders, save_encoders


def main():
    # 1. Load data
    df = pd.read_csv(config.JANUARY_DATA_PATH)

    # 2. Encode categorical columns
    df, topic_encoder, day_encoder = fit_encoders(df)

    # 3. Split features/target
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    # 4. Train
    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model V1 (January) accuracy on January test data: {accuracy:.4f}")

    # 6. Save model + encoders
    joblib.dump(model, config.MODEL_PATH)
    save_encoders(topic_encoder, day_encoder)
    print(f"Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
