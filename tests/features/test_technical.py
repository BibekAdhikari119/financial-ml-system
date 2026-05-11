"""
tests/features/test_technical.py — Unit tests for technical indicator functions.

Uses a synthetic OHLCV DataFrame with 200 rows (seed=42, realistic price
series via cumulative sum) to verify correctness of each indicator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.technical import (
    add_atr,
    add_bollinger_bands,
    add_ema,
    add_macd,
    add_obv,
    add_rsi,
    add_sma,
    build_features,
)


# --------------------------------------------------------------------------- #
# Shared fixture                                                                #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def ohlcv_df() -> pd.DataFrame:
    """
    Synthetic OHLCV DataFrame with 200 rows.

    Prices follow a cumulative-sum random walk (seed=42) starting at 100,
    ensuring realistic positive close values.
    """
    rng = np.random.default_rng(42)
    n = 200

    returns = rng.normal(0.0, 0.5, size=n)
    close = 100.0 + np.cumsum(returns)
    close = np.clip(close, 1.0, None)  # ensure all positive

    # Build OHLC from close with realistic intraday spread
    noise_hi = rng.uniform(0.1, 1.0, size=n)
    noise_lo = rng.uniform(0.1, 1.0, size=n)
    high = close + noise_hi
    low = np.clip(close - noise_lo, 0.01, None)
    open_ = close + rng.uniform(-0.5, 0.5, size=n)

    volume = rng.integers(100_000, 10_000_000, size=n).astype(float)

    index = pd.date_range("2022-01-03", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )
    df.index.name = "Date"
    return df


# --------------------------------------------------------------------------- #
# SMA tests                                                                     #
# --------------------------------------------------------------------------- #

def test_sma_columns_present(ohlcv_df: pd.DataFrame) -> None:
    """add_sma must append sma_10, sma_20, sma_50 columns."""
    result = add_sma(ohlcv_df)
    for period in [10, 20, 50]:
        assert f"sma_{period}" in result.columns, f"sma_{period} missing"


def test_sma_no_mutation(ohlcv_df: pd.DataFrame) -> None:
    """add_sma must not mutate the input DataFrame."""
    original_cols = list(ohlcv_df.columns)
    add_sma(ohlcv_df)
    assert list(ohlcv_df.columns) == original_cols


# --------------------------------------------------------------------------- #
# RSI range test (required)                                                     #
# --------------------------------------------------------------------------- #

def test_rsi_range(ohlcv_df: pd.DataFrame) -> None:
    """rsi_14 values must all be in [0, 100]."""
    result = add_rsi(ohlcv_df)
    rsi_col = result["rsi_14"].dropna()
    assert not rsi_col.empty, "RSI column is entirely NaN"
    assert (rsi_col >= 0.0).all(), "RSI values below 0 found"
    assert (rsi_col <= 100.0).all(), "RSI values above 100 found"


# --------------------------------------------------------------------------- #
# MACD columns test (required)                                                  #
# --------------------------------------------------------------------------- #

def test_macd_columns(ohlcv_df: pd.DataFrame) -> None:
    """add_macd must produce macd_line, macd_signal, and macd_histogram."""
    result = add_macd(ohlcv_df)
    for col in ["macd_line", "macd_signal", "macd_histogram"]:
        assert col in result.columns, f"MACD column '{col}' missing"


def test_macd_histogram_is_difference(ohlcv_df: pd.DataFrame) -> None:
    """macd_histogram must equal macd_line - macd_signal."""
    result = add_macd(ohlcv_df).dropna()
    diff = (result["macd_line"] - result["macd_signal"]).round(10)
    histogram = result["macd_histogram"].round(10)
    pd.testing.assert_series_equal(diff, histogram, check_names=False)


# --------------------------------------------------------------------------- #
# Bollinger band ordering test (required)                                       #
# --------------------------------------------------------------------------- #

def test_bollinger_ordering(ohlcv_df: pd.DataFrame) -> None:
    """bb_upper must be strictly greater than bb_lower for all non-NaN rows."""
    result = add_bollinger_bands(ohlcv_df)
    valid = result.dropna(subset=["bb_upper", "bb_lower"])
    assert not valid.empty, "All Bollinger band values are NaN"
    assert (valid["bb_upper"] > valid["bb_lower"]).all(), (
        "Bollinger band violation: bb_upper <= bb_lower on some rows"
    )


# --------------------------------------------------------------------------- #
# ATR non-negative test (required)                                              #
# --------------------------------------------------------------------------- #

def test_atr_nonnegative(ohlcv_df: pd.DataFrame) -> None:
    """atr_14 must be >= 0 for all non-NaN rows."""
    result = add_atr(ohlcv_df)
    valid = result["atr_14"].dropna()
    assert not valid.empty, "ATR column is entirely NaN"
    assert (valid >= 0.0).all(), "Negative ATR values found"


# --------------------------------------------------------------------------- #
# build_features no-NaN test (required)                                         #
# --------------------------------------------------------------------------- #

def test_build_features_no_nan(ohlcv_df: pd.DataFrame) -> None:
    """build_features output must have zero NaN rows."""
    result = build_features(ohlcv_df)
    assert not result.empty, "build_features returned an empty DataFrame"
    assert result.isna().sum().sum() == 0, (
        f"build_features output contains NaN values:\n"
        f"{result.isna().sum()[result.isna().sum() > 0]}"
    )


# --------------------------------------------------------------------------- #
# OBV test                                                                      #
# --------------------------------------------------------------------------- #

def test_obv_column_present(ohlcv_df: pd.DataFrame) -> None:
    """add_obv must append an 'obv' column."""
    result = add_obv(ohlcv_df)
    assert "obv" in result.columns, "'obv' column missing from add_obv output"


def test_obv_no_nan(ohlcv_df: pd.DataFrame) -> None:
    """OBV must have no NaN values (it is a cumulative sum)."""
    result = add_obv(ohlcv_df)
    assert result["obv"].isna().sum() == 0, "OBV contains NaN values"


# --------------------------------------------------------------------------- #
# EMA test                                                                      #
# --------------------------------------------------------------------------- #

def test_ema_columns_present(ohlcv_df: pd.DataFrame) -> None:
    """add_ema must append ema_12 and ema_26 columns."""
    result = add_ema(ohlcv_df)
    for period in [12, 26]:
        assert f"ema_{period}" in result.columns, f"ema_{period} missing"


# --------------------------------------------------------------------------- #
# Immutability checks across all functions                                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "func",
    [
        add_sma,
        add_ema,
        add_rsi,
        add_macd,
        add_bollinger_bands,
        add_atr,
        add_obv,
        build_features,
    ],
    ids=[
        "add_sma",
        "add_ema",
        "add_rsi",
        "add_macd",
        "add_bollinger_bands",
        "add_atr",
        "add_obv",
        "build_features",
    ],
)
def test_no_input_mutation(ohlcv_df: pd.DataFrame, func) -> None:  # type: ignore[type-arg]
    """Every indicator function must return a new DataFrame without mutating input."""
    original_cols = set(ohlcv_df.columns)
    original_shape = ohlcv_df.shape
    func(ohlcv_df)
    assert set(ohlcv_df.columns) == original_cols, (
        f"{func.__name__} mutated input columns"
    )
    assert ohlcv_df.shape == original_shape, (
        f"{func.__name__} changed input shape"
    )
