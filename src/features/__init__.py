"""
src/features — public re-exports for the feature engineering layer.
"""
from src.features.technical import (
    add_sma,
    add_ema,
    add_rsi,
    add_macd,
    add_bollinger_bands,
    add_atr,
    add_obv,
    build_features,
)
from src.features.sentiment_features import (
    aggregate_daily_sentiment,
    merge_sentiment_features,
)
from src.features.dataset import (
    TimeSeriesDataset,
    create_train_val_test_splits,
)

__all__ = [
    "add_sma",
    "add_ema",
    "add_rsi",
    "add_macd",
    "add_bollinger_bands",
    "add_atr",
    "add_obv",
    "build_features",
    "aggregate_daily_sentiment",
    "merge_sentiment_features",
    "TimeSeriesDataset",
    "create_train_val_test_splits",
]
