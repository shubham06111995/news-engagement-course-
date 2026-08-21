"""
src/train_mlflow.py
Same training as train.py, but logs every run to MLflow so we can
track and compare experiments over time.
Run from the project root as: python src/train_mlflow.py
"""

import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

import config
from src.preprocessing import fit_encoders, save_encoders

mlflow.set_tracking_uri("http://127.0.0.1:5000")


def main():
    df = pd.read_csv(config.JANUARY_DATA_PATH)
    df, topic_encoder, day_encoder = fit_encoders(df)

    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    with mlflow.start_run():
        mlflow.log_param("n_estimators", config.N_ESTIMATORS)

        model = RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model V1 (January) accuracy: {accuracy:.4f}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, name="model")

        # Also save locally, same as train.py does, so predict.py still works
        import joblib
        joblib.dump(model, config.MODEL_PATH)
        save_encoders(topic_encoder, day_encoder)

        print("Run logged to MLflow, model saved locally.")


if __name__ == "__main__":
    main()
