"""
tests/api/test_routes.py — Integration tests for the Financial ML API routes.

All external I/O (fetch_market_data, build_features) is mocked to avoid
real network calls and to produce deterministic, controlled DataFrames.
"""
from __future__ import annotations

import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app

# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N_ROWS = 100
_DATES = pd.date_range("2023-01-01", periods=_N_ROWS, freq="D")

# Base prices built from a random-walk so Close is always positive.
_BASE_CLOSE = 150.0 + np.cumsum(_RNG.normal(0, 1, _N_ROWS))
_BASE_CLOSE = np.abs(_BASE_CLOSE) + 10.0  # guarantee > 0


def _make_ohlcv_df() -> pd.DataFrame:
    """Return a 100-row synthetic OHLCV DataFrame with DatetimeIndex (seed=42)."""
    rng = np.random.default_rng(42)
    close = _BASE_CLOSE.copy()
    high = close * (1.0 + rng.uniform(0.001, 0.02, _N_ROWS))
    low = close * (1.0 - rng.uniform(0.001, 0.02, _N_ROWS))
    open_ = close * (1.0 + rng.uniform(-0.01, 0.01, _N_ROWS))
    volume = rng.integers(1_000_000, 10_000_000, _N_ROWS).astype(float)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=_DATES,
    )


def _make_featured_df() -> pd.DataFrame:
    """Return the OHLCV DataFrame with 3 synthetic feature columns appended."""
    df = _make_ohlcv_df()
    rng = np.random.default_rng(42)
    df["sma_10"] = df["Close"].rolling(10).mean().fillna(df["Close"])
    df["rsi_14"] = 50.0 + rng.uniform(-10, 10, len(df))
    df["macd_line"] = rng.uniform(-1, 1, len(df))
    return df


# ---------------------------------------------------------------------------
# TestClient
# ---------------------------------------------------------------------------

client = TestClient(app)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health_check() -> None:
    """GET /health should return 200 with status=ok and version=1.0.0."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "1.0.0"


@patch("src.api.routes.predict.build_features", return_value=_make_featured_df())
@patch("src.api.routes.predict.fetch_market_data", return_value=_make_ohlcv_df())
def test_predict_200(
    mock_fetch: object,
    mock_features: object,
) -> None:
    """POST /api/v1/predict with valid payload should return 200 and PredictResponse fields."""
    payload = {
        "ticker": "AAPL",
        "start": "2023-01-01",
        "end": "2023-12-31",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "ticker" in body
    assert body["ticker"] == "AAPL"
    assert "signal" in body
    assert "confidence" in body
    assert "timestamp" in body
    # Confidence should be non-negative (abs of signal).
    assert body["confidence"] >= 0.0
    # Timestamp should be a parseable ISO date string.
    datetime.date.fromisoformat(body["timestamp"])


@patch("src.api.routes.backtest.build_features", return_value=_make_featured_df())
@patch("src.api.routes.backtest.fetch_market_data", return_value=_make_ohlcv_df())
def test_backtest_200(
    mock_fetch: object,
    mock_features: object,
) -> None:
    """POST /api/v1/backtest with valid payload should return 200 with required fields."""
    payload = {
        "ticker": "AAPL",
        "start": "2023-01-01",
        "end": "2023-12-31",
        "initial_capital": 100000.0,
        "transaction_cost_bps": 10,
        "slippage_bps": 5,
    }
    response = client.post("/api/v1/backtest", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "metrics" in body
    assert "equity_curve" in body
    assert "dates" in body
    assert isinstance(body["metrics"], dict)
    assert isinstance(body["equity_curve"], list)
    assert isinstance(body["dates"], list)
    assert len(body["equity_curve"]) == len(body["dates"])
    # Equity curve should have at least one value.
    assert len(body["equity_curve"]) > 0
    # Standard metric keys should be present.
    expected_metric_keys = {"sharpe_ratio", "sortino_ratio", "max_drawdown", "cagr", "calmar_ratio"}
    assert expected_metric_keys.issubset(set(body["metrics"].keys()))
    # Benchmark fields should be present and well-formed.
    assert "benchmark_metrics" in body
    assert "benchmark_equity_curve" in body
    assert "benchmark_dates" in body
    assert isinstance(body["benchmark_metrics"], dict)
    assert isinstance(body["benchmark_equity_curve"], list)
    assert isinstance(body["benchmark_dates"], list)
    assert len(body["benchmark_equity_curve"]) > 0
    assert len(body["benchmark_equity_curve"]) == len(body["benchmark_dates"])


@patch("src.api.routes.portfolio.build_features", return_value=_make_featured_df())
@patch("src.api.routes.portfolio.fetch_market_data", return_value=_make_ohlcv_df())
def test_portfolio_200(
    mock_fetch: object,
    mock_features: object,
) -> None:
    """POST /api/v1/portfolio with valid payload should return 200 and blended metrics."""
    payload = {
        "tickers": ["AAPL", "MSFT"],
        "weights": [0.6, 0.4],
        "start": "2023-01-01",
        "end": "2023-12-31",
    }
    response = client.post("/api/v1/portfolio", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tickers"] == ["AAPL", "MSFT"]
    assert body["weights"] == [0.6, 0.4]
    assert isinstance(body["metrics"], dict)
    assert len(body["metrics"]) > 0


def test_portfolio_422_bad_weights() -> None:
    """POST /api/v1/portfolio with weights that don't sum to 1.0 should return 422."""
    payload = {
        "tickers": ["AAPL", "MSFT"],
        "weights": [0.5, 0.3],  # sums to 0.8, not 1.0
        "start": "2023-01-01",
        "end": "2023-12-31",
    }
    response = client.post("/api/v1/portfolio", json=payload)
    assert response.status_code == 422, response.text


def test_predict_422_invalid_dates() -> None:
    """POST /api/v1/predict with end <= start should return 422."""
    payload = {
        "ticker": "AAPL",
        "start": "2023-12-31",
        "end": "2023-01-01",  # end is before start
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422, response.text
