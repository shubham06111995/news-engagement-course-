"""
tests/test_predict.py
Automated tests for the prediction pipeline.
This is what GitHub Actions runs automatically on every push.
"""

import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from src.predict import predict


def test_predict_returns_valid_labels():
    """Predictions should only ever be 'High' or 'Low' - nothing else."""
    sample = pd.DataFrame([{
        "topic": "Technology", "title_length": 42, "article_length": 1100,
        "publish_hour": 10, "day_of_week": "Tue", "is_trending": 1
    }])
    result = predict(sample)
    assert result[0] in ["High", "Low"]


def test_predict_returns_correct_number_of_results():
    """If we give it 2 articles, we should get exactly 2 predictions back."""
    sample = pd.DataFrame([
        {"topic": "Technology", "title_length": 42, "article_length": 1100,
         "publish_hour": 10, "day_of_week": "Tue", "is_trending": 1},
        {"topic": "Sports", "title_length": 30, "article_length": 650,
         "publish_hour": 19, "day_of_week": "Sat", "is_trending": 0},
    ])
    result = predict(sample)
    assert len(result) == 2


def test_technology_trending_predicts_high():
    """A known clear-cut case: Technology + trending should predict High
    (this is the strongest pattern in our January training data)."""
    sample = pd.DataFrame([{
        "topic": "Technology", "title_length": 45, "article_length": 1200,
        "publish_hour": 10, "day_of_week": "Mon", "is_trending": 1
    }])
    result = predict(sample)
    assert result[0] == "High"
