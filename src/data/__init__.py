"""
src/data — public re-exports for the data layer.
"""
from src.data.market import fetch_market_data
from src.data.sentiment import fetch_sentiment_data
from src.data.pipeline import build_pipeline

__all__ = ["fetch_market_data", "fetch_sentiment_data", "build_pipeline"]
