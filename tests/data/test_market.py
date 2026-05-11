"""
tests/data/test_market.py — pytest tests for fetch_market_data.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.market import fetch_market_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATES = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])


def _make_ohlcv(dates: list | pd.DatetimeIndex = _DATES) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with tz-naive DatetimeIndex."""
    n = len(dates)
    df = pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [105.0] * n,
            "Low": [99.0] * n,
            "Close": [102.0] * n,
            "Volume": [1_000_000] * n,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFetchReturnsOHLCVSchema:
    """fetch_market_data must return a DataFrame with the 5 canonical columns."""

    def test_fetch_returns_ohlcv_schema(self, tmp_path: Path) -> None:
        mock_df = _make_ohlcv()
        with patch("yfinance.download", return_value=mock_df):
            result = fetch_market_data("AAPL", "2024-01-01", "2024-01-05", cache_dir=tmp_path)

        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"], (
            f"Expected OHLCV columns, got {list(result.columns)}"
        )


class TestCacheHitSkipsNetwork:
    """Second call must use cache without hitting yfinance when cache is fresh."""

    def test_cache_hit_skips_network(self, tmp_path: Path) -> None:
        mock_df = _make_ohlcv()

        # First call: populate the cache
        with patch("yfinance.download", return_value=mock_df):
            fetch_market_data("AAPL", "2024-01-01", "2024-01-05", cache_dir=tmp_path)

        # Second call: yfinance.download should NOT be invoked
        def _should_not_be_called(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("yfinance.download was called on a cache hit — this is a bug!")

        with patch("yfinance.download", side_effect=_should_not_be_called):
            result = fetch_market_data("AAPL", "2024-01-01", "2024-01-05", cache_dir=tmp_path)

        assert not result.empty, "Cache-hit result should not be empty"


class TestEmptyTickerRaises:
    """fetch_market_data must raise ValueError when yfinance returns empty data."""

    def test_empty_ticker_raises(self, tmp_path: Path) -> None:
        empty_df = pd.DataFrame()
        with patch("yfinance.download", return_value=empty_df):
            with pytest.raises(ValueError, match="no data"):
                fetch_market_data(
                    "INVALID_TICKER_XYZ",
                    "2024-01-01",
                    "2024-01-05",
                    cache_dir=tmp_path,
                )


class TestIndexIsSortedAscending:
    """Returned DataFrame must have a monotonically increasing DatetimeIndex."""

    def test_index_is_sorted_ascending(self, tmp_path: Path) -> None:
        # Supply dates in reverse order to verify sorting
        reversed_dates = _DATES[::-1]
        mock_df = _make_ohlcv(reversed_dates)
        with patch("yfinance.download", return_value=mock_df):
            result = fetch_market_data("AAPL", "2024-01-01", "2024-01-05", cache_dir=tmp_path)

        assert result.index.is_monotonic_increasing, (
            "DatetimeIndex must be sorted ascending"
        )


class TestCacheExpiry:
    """Stale cache (> 24 h old) must trigger a fresh yfinance fetch."""

    def test_stale_cache_triggers_refetch(self, tmp_path: Path) -> None:
        mock_df = _make_ohlcv()

        # Populate the cache
        with patch("yfinance.download", return_value=mock_df):
            fetch_market_data("AAPL", "2024-01-01", "2024-01-05", cache_dir=tmp_path)

        # Wind back the cache file's mtime by 25 hours
        cache_file = tmp_path / "AAPL" / "2024-01-01_2024-01-05_1d.parquet"
        assert cache_file.exists(), "Cache file should have been created"
        stale_mtime = time.time() - (25 * 3600)
        import os
        os.utime(cache_file, (stale_mtime, stale_mtime))

        fetch_count = {"n": 0}

        def _counting_download(*args, **kwargs):  # type: ignore[no-untyped-def]
            fetch_count["n"] += 1
            return mock_df

        with patch("yfinance.download", side_effect=_counting_download):
            fetch_market_data("AAPL", "2024-01-01", "2024-01-05", cache_dir=tmp_path)

        assert fetch_count["n"] == 1, "Expected exactly one yfinance call for stale cache"


class TestMultiLevelColumns:
    """fetch_market_data must handle yfinance multi-level (MultiIndex) columns."""

    def test_multilevel_columns_normalized(self, tmp_path: Path) -> None:
        mock_df = _make_ohlcv()
        # Simulate yfinance MultiIndex columns: (price_level, ticker_level)
        mock_df.columns = pd.MultiIndex.from_tuples(
            [("Open", "AAPL"), ("High", "AAPL"), ("Low", "AAPL"),
             ("Close", "AAPL"), ("Volume", "AAPL")]
        )
        with patch("yfinance.download", return_value=mock_df):
            result = fetch_market_data("AAPL", "2024-01-01", "2024-01-05", cache_dir=tmp_path)

        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
