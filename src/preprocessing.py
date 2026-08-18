"""
src/preprocessing.py
Shared data preparation logic used by BOTH training and prediction,
so encoding rules only need to be defined once, in one place.
"""

import joblib
from sklearn.preprocessing import LabelEncoder
import config


def fit_encoders(df):
    """Create and fit new encoders on training data. Used only during training."""
    topic_encoder = LabelEncoder()
    day_encoder = LabelEncoder()
    df["topic_encoded"] = topic_encoder.fit_transform(df["topic"])
    df["day_encoded"] = day_encoder.fit_transform(df["day_of_week"])
    return df, topic_encoder, day_encoder


def load_encoders():
    """Load previously saved encoders. Used during prediction on new data."""
    topic_encoder = joblib.load(config.TOPIC_ENCODER_PATH)
    day_encoder = joblib.load(config.DAY_ENCODER_PATH)
    return topic_encoder, day_encoder


def apply_encoders(df, topic_encoder, day_encoder):
    """Apply already-fitted encoders to new data (does not refit them)."""
    df["topic_encoded"] = topic_encoder.transform(df["topic"])
    df["day_encoded"] = day_encoder.transform(df["day_of_week"])
    return df


def save_encoders(topic_encoder, day_encoder):
    joblib.dump(topic_encoder, config.TOPIC_ENCODER_PATH)
    joblib.dump(day_encoder, config.DAY_ENCODER_PATH)
