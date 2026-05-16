"""
tests/backtesting/test_metrics.py — Unit tests for src/backtesting/metrics.py.

All tests use known, analytically-derivable inputs so that expected values can
be computed by hand or via straightforward formulas.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtesting.metrics import (
    cagr,
    calmar_ratio,
    compute_all_metrics,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


def test_sharpe_all_zeros() -> None:
    """All-zero returns → std == 0 → Sharpe must be 0.0."""
    result = sharpe_ratio(pd.Series([0.0] * 10))
    assert result == 0.0


def test_sharpe_empty_series() -> None:
    """Empty series must return 0.0 without raising."""
    assert sharpe_ratio(pd.Series([], dtype=float)) == 0.0


def test_sharpe_positive_known_value() -> None:
    """Constant daily return of 0.01 should produce a large positive Sharpe."""
    returns = pd.Series([0.01] * 252)
    result = sharpe_ratio(returns, risk_free_rate=0.0)
    # mean = 0.01, std ≈ 0, but since all values are identical std(ddof=1) = 0 → 0.0
    # Use a slight perturbation so std > 0.
    rng = np.random.default_rng(0)
    returns_noisy = pd.Series(0.01 + rng.normal(0, 0.001, 252))
    result_noisy = sharpe_ratio(returns_noisy, risk_free_rate=0.0)
    assert result_noisy > 5.0  # very high Sharpe for low-noise positive drift


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_monotonic_increase() -> None:
    """Strictly increasing returns → equity never below prior peak → drawdown = 0.0."""
    returns = pd.Series([0.005] * 50)  # every day up 0.5%
    result = max_drawdown(returns)
    assert result == 0.0


def test_max_drawdown_known_drop() -> None:
    """50% single-day drop after several up days → max drawdown ≈ -0.5."""
    # Build: 10 days of 0% then one day of -50%.
    returns = pd.Series([0.0] * 10 + [-0.5])
    result = max_drawdown(returns)
    assert math.isclose(result, -0.5, rel_tol=1e-6)


def test_max_drawdown_empty_series() -> None:
    assert max_drawdown(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------


def test_sortino_only_penalizes_downside() -> None:
    """Sortino > Sharpe when large upside spikes inflate total std but not downside std.

    Construct a series where large positive outliers drive up overall std
    (hurting Sharpe) while the downside is modest and varied — so Sortino's
    smaller denominator yields a higher ratio than Sharpe.
    """
    rng = np.random.default_rng(42)
    # Base: small positive drift with genuine downside variance
    base = rng.normal(0.002, 0.005, 200)
    # Add large positive spikes to inflate overall std without adding downside risk
    base[::10] = 0.05  # every 10th day: +5% spike
    returns = pd.Series(base)

    sr = sharpe_ratio(returns, risk_free_rate=0.0)
    so = sortino_ratio(returns, risk_free_rate=0.0)

    # Large upside spikes inflate total std → Sharpe penalised; Sortino ignores upside
    assert so > sr, f"Expected sortino ({so:.4f}) > sharpe ({sr:.4f})"


def test_sortino_all_positive_returns() -> None:
    """All returns above rf → no downside deviation → Sortino is unbounded (inf)."""
    returns = pd.Series([0.01] * 20)
    result = sortino_ratio(returns, risk_free_rate=0.0)
    assert math.isinf(result) and result > 0


# ---------------------------------------------------------------------------
# cagr
# ---------------------------------------------------------------------------


def test_cagr_known_value() -> None:
    """1% daily return for exactly 252 days → CAGR ≈ (1.01^252 - 1)."""
    returns = pd.Series([0.01] * 252)
    result = cagr(returns, periods_per_year=252)
    expected = (1.01 ** 252) - 1.0
    assert math.isclose(result, expected, rel_tol=1e-6), (
        f"Expected ~{expected:.4f}, got {result:.4f}"
    )


def test_cagr_zero_returns() -> None:
    """All-zero returns → CAGR must be 0.0 (no growth)."""
    returns = pd.Series([0.0] * 100)
    result = cagr(returns)
    assert math.isclose(result, 0.0, abs_tol=1e-10)


def test_cagr_empty_series() -> None:
    assert cagr(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------


def test_calmar_zero_drawdown() -> None:
    """No drawdown → equity only rose → Calmar is unbounded (inf)."""
    returns = pd.Series([0.005] * 50)
    result = calmar_ratio(returns)
    assert math.isinf(result) and result > 0


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


def test_compute_all_metrics_keys() -> None:
    """Returned dict must contain exactly the 5 expected metric keys."""
    expected_keys = {"sharpe_ratio", "sortino_ratio", "max_drawdown", "cagr", "calmar_ratio"}
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.001, 0.01, 252))
    result = compute_all_metrics(returns)
    assert set(result.keys()) == expected_keys


def test_compute_all_metrics_types() -> None:
    """All metric values must be Python floats (not numpy scalars)."""
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0005, 0.005, 100))
    metrics = compute_all_metrics(returns)
    for key, value in metrics.items():
        assert isinstance(value, float), f"Metric '{key}' is {type(value)}, expected float"


def test_compute_all_metrics_empty() -> None:
    """All metrics must be 0.0 for an empty series."""
    metrics = compute_all_metrics(pd.Series([], dtype=float))
    for key, value in metrics.items():
        assert value == 0.0, f"Expected 0.0 for '{key}', got {value}"
