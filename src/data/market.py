"""
src/data/market.py — yfinance OHLCV fetcher with disk-based Parquet caching.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "financial_ml" / "market"
_CACHE_MAX_AGE_SECONDS: int = 86_400  # 24 hours


def _safe_ticker(ticker: str) -> str:
    """Return a filesystem-safe version of *ticker* by replacing any character
    that is not an uppercase letter, digit, dot, underscore, caret, equals sign,
    or hyphen with an underscore.  This prevents path-traversal attacks such as
    ``../../etc/passwd`` while preserving common ticker formats (e.g. BRK.B,
    ^GSPC, ES=F).
    """
    return re.sub(r"[^A-Z0-9._^=\-]", "_", ticker.upper())


def _cache_path(
    cache_dir: Path,
    ticker: str,
    start: str,
    end: str,
    interval: str,
) -> Path:
    """Return the Parquet cache file path for the given parameters.

    The ticker is sanitised via :func:`_safe_ticker` before being embedded in
    the path so that adversarial inputs cannot traverse outside *cache_dir*.
    The original (unsanitised) ticker is still used for the yfinance download
    call in :func:`fetch_market_data`.
    """
    safe = _safe_ticker(ticker)
    return cache_dir / safe / f"{start}_{end}_{interval}.parquet"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten multi-level columns produced by yfinance and keep only OHLCV.

    yfinance >= 0.2.x can return a MultiIndex like (Price, Ticker) or
    a plain Index like ['Open', 'High', ...].  We normalize both.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Drop the ticker level; keep the price level (level 0)
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # Standardize to title-case OHLCV names
    rename_map: dict[str, str] = {}
    for col in df.columns:
        col_lower = str(col).lower()
        for canonical in ("Open", "High", "Low", "Close", "Volume"):
            if col_lower == canonical.lower():
                rename_map[col] = canonical
                break
    df = df.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"yfinance response is missing expected columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df[required]


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the DatetimeIndex to UTC-normalized, tz-naive, sorted ascending."""
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    else:
        # Attempt to interpret naive timestamps as UTC
        idx = idx.tz_localize("UTC").tz_localize(None)
    df.index = idx
    df.index.name = "Date"
    return df.sort_index(ascending=True)


def _is_cache_valid(cache_file: Path) -> bool:
    """Return True if the cache file exists and is less than 24 hours old."""
    if not cache_file.exists():
        return False
    age = time.time() - cache_file.stat().st_mtime
    return age < _CACHE_MAX_AGE_SECONDS


def fetch_market_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV price data for *ticker* between *start* and *end*.

    Parameters
    ----------
    ticker:
        Equity ticker symbol, e.g. ``"AAPL"``.
    start:
        Inclusive start date in ``YYYY-MM-DD`` format.
    end:
        Exclusive end date in ``YYYY-MM-DD`` format.
    interval:
        yfinance interval string, e.g. ``"1d"``, ``"1h"``.  Defaults to ``"1d"``.
    cache_dir:
        Root directory for Parquet cache files.  Defaults to
        ``~/.cache/financial_ml/market``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["Open", "High", "Low", "Close", "Volume"]``
        and a UTC-normalized, tz-naive ``DatetimeIndex`` named ``"Date"``,
        sorted ascending.

    Raises
    ------
    ValueError
        If yfinance returns an empty DataFrame for the requested parameters.
    """
    resolved_cache_dir: Path = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR
    cache_file = _cache_path(resolved_cache_dir, ticker, start, end, interval)

    if _is_cache_valid(cache_file):
        logger.debug("Cache hit for %s (%s → %s, %s)", ticker, start, end, interval)
        df = pd.read_parquet(cache_file)
        return df

    logger.debug("Cache miss — fetching %s (%s → %s, %s) via yfinance", ticker, start, end, interval)
    raw: pd.DataFrame = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for ticker='{ticker}', "
            f"start='{start}', end='{end}', interval='{interval}'."
        )

    df = _normalize_columns(raw)
    df = _normalize_index(df)

    # Persist to cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)
    logger.debug("Wrote cache file %s", cache_file)

    return df
