"""
config.py
Central place for all file paths and settings used across the project.
Changing a path or setting here updates it everywhere - no need to
hunt through multiple scripts.
"""

# Data paths
JANUARY_DATA_PATH = "data/raw/articles_january.csv"
MARCH_DATA_PATH = "data/raw/articles_march.csv"

# Model artifact paths
MODEL_PATH = "models/model_v1.pkl"
TOPIC_ENCODER_PATH = "models/topic_encoder.pkl"
DAY_ENCODER_PATH = "models/day_encoder.pkl"

# Feature columns used by the model
FEATURE_COLUMNS = [
    "topic_encoded", "title_length", "article_length",
    "publish_hour", "day_encoded", "is_trending"
]
TARGET_COLUMN = "engagement"

# Training settings
TEST_SIZE = 0.25
RANDOM_STATE = 42
N_ESTIMATORS = 100
