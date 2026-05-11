"""
src/features/sentiment_features.py — Sentiment aggregation and market merge.

Provides two public functions:
- ``aggregate_daily_sentiment``: collapses intraday sentiment rows to a daily
  summary with mean, std, count, and EWM statistics.
- ``merge_sentiment_features``: left-joins aggregated sentiment onto a market
  DataFrame, forward-filling up to 3 trading days and zeroing the rest.

No forward leakage: sentiment for day *t* only uses articles published on or
before day *t*.
"""
from __future__ import annotations

import pandas as pd


def aggregate_daily_sentiment(
    df: pd.DataFrame,
    score_col: str = "sentiment_score",
) -> pd.DataFrame:
    """
    Aggregate intraday sentiment rows to daily summary statistics.

    Parameters
    ----------
    df:
        DataFrame with a DatetimeIndex (any intraday granularity) and at
        least a ``score_col`` column containing numeric sentiment scores.
    score_col:
        Name of the column holding the raw sentiment score.
        Defaults to ``"sentiment_score"``.

    Returns
    -------
    pd.DataFrame
        Daily DataFrame (midnight DatetimeIndex) with columns:
        ``["sentiment_mean", "sentiment_std", "sentiment_count",
        "sentiment_ewm"]``.

    Notes
    -----
    - ``sentiment_ewm`` is the 24-period exponentially-weighted mean of
      *score_col*, computed *before* daily grouping so that intraday
      ordering contributes to the decay.
    - Rows in *df* are treated as occurring at the given timestamp; grouping
      by date normalises the index to midnight (no timezone shift beyond
      what the caller provides).
    """
    if score_col not in df.columns:
        raise ValueError(
            f"Column '{score_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    work = df[[score_col]].copy()
    # Compute EWM on the full intraday series before grouping
    work["_ewm"] = (
        work[score_col].ewm(span=24, adjust=False).mean()
    )
    # Normalize index to date (midnight)
    work.index = work.index.normalize()
    work.index.name = "date"

    daily = work.groupby(work.index).agg(
        sentiment_mean=(score_col, "mean"),
        sentiment_std=(score_col, "std"),
        sentiment_count=(score_col, "count"),
        sentiment_ewm=("_ewm", "last"),
    )
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    return daily


def merge_sentiment_features(
    market_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join aggregated sentiment onto *market_df* without forward leakage.

    Sentiment for trading day *t* only uses articles published on or before
    day *t* (ensured by ``aggregate_daily_sentiment`` grouping on publication
    date and a left join that never introduces future observations).

    Parameters
    ----------
    market_df:
        OHLCV (or feature-enriched) DataFrame with a DatetimeIndex.
    sentiment_df:
        Daily sentiment DataFrame as returned by
        :func:`aggregate_daily_sentiment`.  If raw (intraday), it is passed
        through ``aggregate_daily_sentiment`` first.

    Returns
    -------
    pd.DataFrame
        *market_df* with sentiment columns appended.  Missing sentiment days
        are forward-filled for up to 3 consecutive trading days; gaps larger
        than 3 days are filled with 0 (neutral).
    """
    SENTIMENT_COLS = [
        "sentiment_mean",
        "sentiment_std",
        "sentiment_count",
        "sentiment_ewm",
    ]

    # ------------------------------------------------------------------
    # Ensure sentiment is aggregated to daily granularity
    # ------------------------------------------------------------------
    if not all(c in sentiment_df.columns for c in SENTIMENT_COLS):
        # Raw intraday sentiment passed — aggregate it first
        score_col = "sentiment_score"
        if score_col not in sentiment_df.columns:
            available = list(sentiment_df.columns)
            # Attempt to detect the score column heuristically
            numeric_cols = sentiment_df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                score_col = numeric_cols[0]
            else:
                raise ValueError(
                    f"Cannot determine sentiment score column from: {available}"
                )
        daily_sentiment = aggregate_daily_sentiment(sentiment_df, score_col=score_col)
    else:
        daily_sentiment = sentiment_df.copy()

    # ------------------------------------------------------------------
    # Normalise market index to midnight for the join key
    # ------------------------------------------------------------------
    market_norm = market_df.copy()
    market_norm.index = market_norm.index.normalize()
    market_norm.index.name = "date"

    daily_sentiment.index = pd.DatetimeIndex(daily_sentiment.index).normalize()
    daily_sentiment.index.name = "date"

    # ------------------------------------------------------------------
    # Left join — keeps all market trading days, pulls in matching sentiment
    # ------------------------------------------------------------------
    merged = market_norm.join(daily_sentiment[SENTIMENT_COLS], how="left")

    # ------------------------------------------------------------------
    # Forward-fill sentiment with a maximum of 3 periods (trading days)
    # ------------------------------------------------------------------
    merged[SENTIMENT_COLS] = merged[SENTIMENT_COLS].ffill(limit=3)

    # Beyond the 3-day limit, fill remaining NaNs with 0 (neutral)
    merged[SENTIMENT_COLS] = merged[SENTIMENT_COLS].fillna(0.0)

    return merged.sort_index(ascending=True)
