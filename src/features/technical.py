"""
src/features/technical.py — Pure-function technical indicator library.

All functions accept a OHLCV DataFrame and return a new DataFrame with
additional columns appended.  Input is never mutated.  All computations
use only past data — no forward-looking (.shift(-n)) operations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise ValueError if any required columns are absent."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Individual indicator functions
# ---------------------------------------------------------------------------

def add_sma(
    df: pd.DataFrame,
    periods: list[int] | None = None,
) -> pd.DataFrame:
    """
    Append Simple Moving Average columns to *df*.

    Parameters
    ----------
    df:
        OHLCV DataFrame.  Must contain a ``Close`` column.
    periods:
        Rolling window lengths.  Defaults to ``[10, 20, 50]``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with columns ``sma_<period>`` appended.
    """
    if periods is None:
        periods = [10, 20, 50]
    _require_columns(df, ["Close"])
    out = df.copy()
    for p in periods:
        out[f"sma_{p}"] = out["Close"].rolling(window=p, min_periods=p).mean()
    return out


def add_ema(
    df: pd.DataFrame,
    periods: list[int] | None = None,
) -> pd.DataFrame:
    """
    Append Exponential Moving Average columns to *df*.

    Parameters
    ----------
    df:
        OHLCV DataFrame.  Must contain a ``Close`` column.
    periods:
        Span values for EWM.  Defaults to ``[12, 26]``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with columns ``ema_<period>`` appended.
    """
    if periods is None:
        periods = [12, 26]
    _require_columns(df, ["Close"])
    out = df.copy()
    for p in periods:
        out[f"ema_{p}"] = (
            out["Close"].ewm(span=p, min_periods=p, adjust=False).mean()
        )
    return out


def add_rsi(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Append Relative Strength Index column to *df* using Wilder's smoothing.

    Uses exponential moving average with ``alpha = 1/period`` and
    ``adjust=False`` to match Wilder's original specification.  Values are
    clipped to [0, 100].

    Parameters
    ----------
    df:
        OHLCV DataFrame.  Must contain a ``Close`` column.
    period:
        Look-back window.  Defaults to ``14``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with column ``rsi_<period>`` appended.
    """
    _require_columns(df, ["Close"])
    out = df.copy()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing: EMA with alpha = 1/period, adjust=False
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # When avg_loss is 0 and avg_gain > 0 → RSI = 100
    rsi = rsi.where(avg_loss != 0.0, other=100.0)
    # When both are 0 → RSI = 50 (neutral)
    rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), other=50.0)

    out[f"rsi_{period}"] = rsi.clip(0.0, 100.0)
    return out


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Append MACD line, signal line, and histogram columns to *df*.

    Parameters
    ----------
    df:
        OHLCV DataFrame.  Must contain a ``Close`` column.
    fast:
        Fast EMA span.  Defaults to ``12``.
    slow:
        Slow EMA span.  Defaults to ``26``.
    signal:
        Signal EMA span applied to the MACD line.  Defaults to ``9``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with columns ``macd_line``, ``macd_signal``,
        and ``macd_histogram`` appended.
    """
    _require_columns(df, ["Close"])
    out = df.copy()

    ema_fast = out["Close"].ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, min_periods=slow, adjust=False).mean()

    out["macd_line"] = ema_fast - ema_slow
    out["macd_signal"] = (
        out["macd_line"].ewm(span=signal, min_periods=signal, adjust=False).mean()
    )
    out["macd_histogram"] = out["macd_line"] - out["macd_signal"]
    return out


def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Append Bollinger Band columns to *df*.

    Parameters
    ----------
    df:
        OHLCV DataFrame.  Must contain a ``Close`` column.
    period:
        Rolling window length for the middle band (SMA).  Defaults to ``20``.
    std_dev:
        Number of standard deviations for the upper/lower bands.
        Defaults to ``2.0``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with columns ``bb_upper``, ``bb_lower``, and
        ``bb_pct_b`` appended.

    Notes
    -----
    ``bb_pct_b = (Close - bb_lower) / (bb_upper - bb_lower)`` —
    measures Close position within the band.  NaN when band width is 0.
    """
    _require_columns(df, ["Close"])
    out = df.copy()

    rolling = out["Close"].rolling(window=period, min_periods=period)
    middle = rolling.mean()
    std = rolling.std(ddof=1)

    out["bb_upper"] = middle + std_dev * std
    out["bb_lower"] = middle - std_dev * std

    band_width = out["bb_upper"] - out["bb_lower"]
    out["bb_pct_b"] = (out["Close"] - out["bb_lower"]) / band_width.replace(0.0, np.nan)
    return out


def add_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Append Average True Range column to *df*.

    ATR is the Wilder-smoothed (EMA alpha=1/period) average of the true range.
    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|).

    Parameters
    ----------
    df:
        OHLCV DataFrame.  Must contain ``High``, ``Low``, and ``Close``.
    period:
        Look-back window.  Defaults to ``14``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with column ``atr_<period>`` appended.  Values are
        guaranteed non-negative.
    """
    _require_columns(df, ["High", "Low", "Close"])
    out = df.copy()

    prev_close = out["Close"].shift(1)
    tr = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    out[f"atr_{period}"] = atr.clip(lower=0.0)
    return out


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append On-Balance Volume column to *df*.

    OBV accumulates volume: add on up-days, subtract on down-days, unchanged
    on flat days.

    Parameters
    ----------
    df:
        OHLCV DataFrame.  Must contain ``Close`` and ``Volume``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with column ``obv`` appended.
    """
    _require_columns(df, ["Close", "Volume"])
    out = df.copy()

    close_diff = out["Close"].diff()
    direction = np.sign(close_diff).fillna(0.0)
    out["obv"] = (direction * out["Volume"]).cumsum()
    return out


# ---------------------------------------------------------------------------
# Composite builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all technical indicators to *df* and return a clean DataFrame.

    Calls ``add_sma``, ``add_ema``, ``add_rsi``, ``add_macd``,
    ``add_bollinger_bands``, ``add_atr``, and ``add_obv`` in sequence.
    After all indicators are appended, drops any rows that contain NaN
    values (caused by look-back warm-up periods).

    Parameters
    ----------
    df:
        OHLCV DataFrame.  Must contain ``Open``, ``High``, ``Low``,
        ``Close``, and ``Volume`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with all indicator columns appended and NaN rows
        removed.  Input is never mutated.
    """
    out = df.copy()
    out = add_sma(out)
    out = add_ema(out)
    out = add_rsi(out)
    out = add_macd(out)
    out = add_bollinger_bands(out)
    out = add_atr(out)
    out = add_obv(out)
    out = out.dropna()
    return out
