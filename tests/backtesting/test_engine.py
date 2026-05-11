"""
tests/backtesting/test_engine.py — Integration tests for BacktestEngine.

Uses a synthetic 60-row OHLCV DataFrame (numpy seed=42) and EnsembleStrategy
with synthetic EnsembleSignal objects.  All tests validate engine mechanics
rather than specific return magnitudes.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestEngine, BacktestResult
from src.backtesting.strategy import EnsembleStrategy
from src.models.ensemble import EnsembleSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_ROWS = 60
_RNG = np.random.default_rng(42)


def _make_ohlcv(n: int = N_ROWS, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")

    # Random walk for close prices, anchored at 100.
    log_returns = rng.normal(0.0005, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.005, n))
    volume = rng.integers(100_000, 1_000_000, n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _make_signals(n: int, value: float = 0.5) -> list[EnsembleSignal]:
    """Create *n* EnsembleSignal objects with a constant ensemble_score."""
    return [
        EnsembleSignal(
            transformer_score=value,
            sentiment_score=value,
            ensemble_score=value,
            confidence=abs(value),
        )
        for _ in range(n)
    ]


def _make_mixed_signals(n: int, seed: int = 42) -> list[EnsembleSignal]:
    """Create *n* EnsembleSignal objects with varied ensemble_score values."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(-0.8, 0.8, n)
    return [
        EnsembleSignal(
            transformer_score=float(s),
            sentiment_score=float(s),
            ensemble_score=float(s),
            confidence=abs(float(s)),
        )
        for s in scores
    ]


# ---------------------------------------------------------------------------
# Test 1: transaction costs reduce final equity
# ---------------------------------------------------------------------------


def test_transaction_costs_reduce_returns() -> None:
    """Running with transaction_cost_bps=10 must produce lower final equity
    than running with transaction_cost_bps=0, given identical signals that
    trigger at least one trade."""
    df = _make_ohlcv()
    signals = _make_mixed_signals(N_ROWS)  # varied signals → many position changes

    result_with_cost = BacktestEngine(
        df, EnsembleStrategy(signals), transaction_cost_bps=10, slippage_bps=0
    ).run()
    result_no_cost = BacktestEngine(
        df, EnsembleStrategy(signals), transaction_cost_bps=0, slippage_bps=0
    ).run()

    assert result_with_cost.equity_curve.iloc[-1] < result_no_cost.equity_curve.iloc[-1], (
        "Equity with costs should be lower than without costs."
    )


# ---------------------------------------------------------------------------
# Test 2: slippage is applied to fill price
# ---------------------------------------------------------------------------


def test_slippage_applied_to_fill_price() -> None:
    """When position changes, fill_price must differ from the raw Open price
    by exactly slippage_bps/10000 * open_price (in the direction of the trade).
    """
    df = _make_ohlcv()
    slip_bps = 5
    signals = _make_signals(N_ROWS, value=0.6)  # constant long signal

    result = BacktestEngine(
        df, EnsembleStrategy(signals), transaction_cost_bps=0, slippage_bps=slip_bps
    ).run()

    # There must be at least one trade (the initial entry).
    assert not result.trades.empty, "Expected at least one trade."

    # For the first trade: position goes from 0 → 0.6 (long), so direction = +1.
    first_trade = result.trades.iloc[0]
    first_trade_date = first_trade["date"]

    # The fill should be at the open of the trade date.
    raw_open = float(df.loc[first_trade_date, "Open"])
    expected_fill = raw_open * (1 + slip_bps / 10_000.0)

    assert math.isclose(first_trade["fill_price"], expected_fill, rel_tol=1e-6), (
        f"fill_price {first_trade['fill_price']:.6f} != expected {expected_fill:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: BacktestResult fields are all populated
# ---------------------------------------------------------------------------


def test_backtest_result_fields_populated() -> None:
    """All fields of BacktestResult must be non-empty / non-None after a run."""
    df = _make_ohlcv()
    signals = _make_mixed_signals(N_ROWS)
    result = BacktestEngine(df, EnsembleStrategy(signals)).run()

    # equity_curve: length == len(df)
    assert isinstance(result.equity_curve, pd.Series)
    assert len(result.equity_curve) == N_ROWS, (
        f"equity_curve length {len(result.equity_curve)} != {N_ROWS}"
    )

    # daily_returns: same length
    assert isinstance(result.daily_returns, pd.Series)
    assert len(result.daily_returns) == N_ROWS

    # trades: DataFrame with correct columns
    assert isinstance(result.trades, pd.DataFrame)
    expected_cols = {"date", "signal", "fill_price", "cost", "pnl"}
    assert expected_cols.issubset(set(result.trades.columns)), (
        f"Missing columns: {expected_cols - set(result.trades.columns)}"
    )

    # metrics: all 5 keys present
    expected_metric_keys = {"sharpe_ratio", "sortino_ratio", "max_drawdown", "cagr", "calmar_ratio"}
    assert expected_metric_keys == set(result.metrics.keys())

    # Initial equity must equal initial_capital
    assert math.isclose(result.equity_curve.iloc[0], 100_000.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Test 4: no-lookahead — signal at t uses price at t+1
# ---------------------------------------------------------------------------


def test_no_lookahead_signal_consumption() -> None:
    """The date recorded in the trades DataFrame must correspond to the bar
    *after* the signal bar (signal at index t → fill at index t+1).

    We use a strategy that flips from 0 to a known signal at a specific index
    and verify the trade is recorded on the *next* date.
    """
    df = _make_ohlcv()
    n = len(df)

    # All signals are 0 except index 5 which is 0.7.
    # This should trigger the first real trade at df.index[6] (the next bar).
    scores = [0.0] * n
    scores[5] = 0.7
    signals = [
        EnsembleSignal(
            transformer_score=s,
            sentiment_score=s,
            ensemble_score=s,
            confidence=abs(s),
        )
        for s in scores
    ]

    result = BacktestEngine(
        df, EnsembleStrategy(signals), transaction_cost_bps=0, slippage_bps=0
    ).run()

    assert not result.trades.empty, "Expected at least one trade."

    # First trade date must be df.index[6], not df.index[5].
    first_trade_date = result.trades.iloc[0]["date"]
    expected_fill_date = df.index[6]

    assert first_trade_date == expected_fill_date, (
        f"Signal at index 5 (date={df.index[5].date()}) should fill at "
        f"index 6 (date={expected_fill_date.date()}), but trade was at "
        f"{first_trade_date}"
    )


# ---------------------------------------------------------------------------
# Test 5: equity curve starts at initial capital
# ---------------------------------------------------------------------------


def test_initial_capital_respected() -> None:
    """First equity_curve value must equal initial_capital regardless of signals."""
    df = _make_ohlcv()
    signals = _make_mixed_signals(N_ROWS)
    capital = 250_000.0
    result = BacktestEngine(df, EnsembleStrategy(signals), initial_capital=capital).run()
    assert math.isclose(result.equity_curve.iloc[0], capital, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Test 6: signal clipping — signals outside [-1, 1] are clipped
# ---------------------------------------------------------------------------


def test_signals_clipped_to_unit_range() -> None:
    """Signals with magnitude > 1.0 must be clipped to [-1, 1] without error."""
    df = _make_ohlcv()
    # Construct signals with out-of-range values.
    scores = [2.0, -3.0] * (N_ROWS // 2)
    signals = [
        EnsembleSignal(
            transformer_score=s,
            sentiment_score=min(max(s, -1.0), 1.0),
            ensemble_score=s,  # intentionally out of range
            confidence=abs(s),
        )
        for s in scores
    ]

    # Must not raise.
    result = BacktestEngine(df, EnsembleStrategy(signals)).run()

    # All recorded signals in trades must be in [-1, 1] after clipping.
    if not result.trades.empty:
        assert (result.trades["signal"].abs() <= 1.0 + 1e-9).all(), (
            "Trade signals exceed clipped range."
        )
