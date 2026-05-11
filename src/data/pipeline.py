"""
src/data/pipeline.py — Unified market + sentiment data pipeline.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from src.data.market import fetch_market_data
from src.data.sentiment import fetch_sentiment_data

logger = logging.getLogger(__name__)


def build_pipeline(
    ticker: str,
    start: str,
    end: str,
    sentiment_query: str | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Build a combined OHLCV + (optional) sentiment DataFrame.

    Parameters
    ----------
    ticker:
        Equity ticker symbol, e.g. ``"AAPL"``.
    start:
        Inclusive start date in ``YYYY-MM-DD`` format.
    end:
        Exclusive end date in ``YYYY-MM-DD`` format.
    sentiment_query:
        If provided and ``NEWS_API_KEY`` is set in the environment, sentiment
        data is fetched for this query and merged with the market data.
    cache_dir:
        Passed through to :func:`fetch_market_data` for Parquet caching.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with a UTC-normalized, tz-naive ``DatetimeIndex``
        sorted ascending.  Price columns are never forward-filled.  Sentiment
        columns (suffixed ``_sentiment``) are forward-filled to cover trading
        days with no news.

    Notes
    -----
    - If sentiment fetch raises *any* exception, a warning is logged and the
      function falls back to market-only data.
    - Sentiment merge uses daily granularity (outer join on date component of
      the index) so intraday market data is supported in the future.
    """
    market_df: pd.DataFrame = fetch_market_data(
        ticker=ticker,
        start=start,
        end=end,
        cache_dir=cache_dir,
    )

    should_fetch_sentiment = (
        sentiment_query is not None
        and bool(os.environ.get("NEWS_API_KEY"))
    )

    if not should_fetch_sentiment:
        return market_df.sort_index(ascending=True)

    # Attempt to fetch and merge sentiment
    try:
        sentiment_df: pd.DataFrame = fetch_sentiment_data(
            query=sentiment_query,  # type: ignore[arg-type]  # guarded above
            from_date=start,
            to_date=end,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sentiment fetch failed for query=%r (%s → %s): %s. "
            "Returning market-only data.",
            sentiment_query,
            start,
            end,
            exc,
        )
        return market_df.sort_index(ascending=True)

    if sentiment_df.empty:
        logger.debug("Sentiment fetch returned zero articles; returning market-only data.")
        return market_df.sort_index(ascending=True)

    # ------------------------------------------------------------------ #
    # Aggregate sentiment to daily granularity                             #
    # Sentiment DataFrame may have multiple rows per day; we keep the last #
    # article's columns per calendar day.  For more sophisticated          #
    # aggregation (e.g. average sentiment score) that belongs in           #
    # src/features/ — not here.                                            #
    # ------------------------------------------------------------------ #
    sentiment_daily = sentiment_df.copy()
    # Normalize index to midnight (date-only) for daily join
    sentiment_daily.index = sentiment_daily.index.normalize()
    # Keep last record per day to avoid duplicates on merge
    sentiment_daily = sentiment_daily[~sentiment_daily.index.duplicated(keep="last")]

    # Rename sentiment columns to avoid collision with OHLCV names
    sentiment_daily.columns = [f"{col}_sentiment" for col in sentiment_daily.columns]

    # Normalize market index to midnight as well for the join key
    market_daily = market_df.copy()
    market_daily.index = market_daily.index.normalize()

    # Outer join so we keep all trading days (some may have no news)
    combined: pd.DataFrame = market_daily.join(sentiment_daily, how="outer")

    # Forward-fill ONLY sentiment columns — never price columns
    sentiment_cols = [c for c in combined.columns if c.endswith("_sentiment")]
    price_cols = [c for c in combined.columns if not c.endswith("_sentiment")]

    if sentiment_cols:
        combined[sentiment_cols] = combined[sentiment_cols].ffill()

    # Keep only rows that have price data (drop pure-sentiment-only rows that
    # have no corresponding trading day)
    combined = combined.dropna(subset=price_cols, how="all")

    return combined.sort_index(ascending=True)
