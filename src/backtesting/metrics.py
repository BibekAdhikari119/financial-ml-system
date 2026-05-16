"""
src/backtesting/metrics.py — Pure performance metric functions for daily return series.

All functions accept a pd.Series of daily returns (not prices/equity values).
An empty Series or all-zero Series returns 0.0 for every metric to allow safe
composition inside compute_all_metrics without guarding the call site.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Annualised Sharpe ratio.

    Formula:
        (mean_excess_return * periods_per_year) / (std * sqrt(periods_per_year))

    Args:
        returns: Daily simple returns series.
        risk_free_rate: Annual risk-free rate (default 2%).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Annualised Sharpe ratio. 0.0 when std == 0 or the series is empty.
    """
    if returns.empty:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    std = excess.std(ddof=1)

    if std < 1e-10 or math.isnan(std):
        return 0.0

    return float((excess.mean() * periods_per_year) / (std * math.sqrt(periods_per_year)))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Annualised Sortino ratio (penalises only downside deviation).

    Downside deviation uses returns below the daily risk-free hurdle rate.

    Args:
        returns: Daily simple returns series.
        risk_free_rate: Annual risk-free rate (default 2%).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Annualised Sortino ratio. 0.0 when downside std == 0 or series is empty.
    """
    if returns.empty:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    downside = excess[excess < 0]

    if downside.empty:
        # No downside observations: all returns beat the hurdle → ratio is unbounded
        return float("inf")

    downside_std = downside.std(ddof=1)

    if downside_std < 1e-10 or math.isnan(downside_std):
        return 0.0

    mean_excess = excess.mean()
    return float((mean_excess * periods_per_year) / (downside_std * math.sqrt(periods_per_year)))


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum drawdown as a negative fraction (or 0.0 for monotonically increasing equity).

    Equity curve is computed as (1 + returns).cumprod().

    Args:
        returns: Daily simple returns series.

    Returns:
        Minimum drawdown as a float <= 0 (e.g. -0.25 means 25% drawdown).
        Returns 0.0 when the series is empty or equity never falls below a prior peak.
    """
    if returns.empty:
        return 0.0

    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    result = float(drawdown.min())

    # Clip tiny floating-point noise to 0.0 for the monotonically increasing case.
    return result if result < 0.0 else 0.0


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compound Annual Growth Rate.

    Formula:
        (final_value / 1.0) ** (periods_per_year / n_periods) - 1

    where final_value = (1 + returns).prod().

    Args:
        returns: Daily simple returns series.
        periods_per_year: Trading days per year (default 252).

    Returns:
        CAGR as a float. 0.0 when the series is empty.
    """
    if returns.empty:
        return 0.0

    n_periods = len(returns)
    final_value = float((1.0 + returns).prod())

    if final_value <= 0:
        return -1.0  # total loss

    return float(final_value ** (periods_per_year / n_periods) - 1.0)


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calmar ratio: CAGR divided by the absolute maximum drawdown.

    Args:
        returns: Daily simple returns series.
        periods_per_year: Trading days per year (default 252).

    Returns:
        Calmar ratio. 0.0 when max_drawdown == 0 or the series is empty.
    """
    if returns.empty:
        return 0.0

    mdd = max_drawdown(returns)
    if mdd == 0.0:
        # Zero drawdown: equity only rose → ratio is unbounded
        return float("inf")

    return float(cagr(returns, periods_per_year) / abs(mdd))


_INF_SENTINEL = 999.0  # JSON-safe substitute for unbounded ratios (inf → null in Pydantic v2)


def compute_all_metrics(returns: pd.Series) -> dict[str, float]:
    """
    Compute the full suite of performance metrics for a daily return series.

    Args:
        returns: Daily simple returns series.

    Returns:
        Dictionary with keys:
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "cagr", "calmar_ratio"

    Note: unbounded ratios (e.g. Sortino with no downside, Calmar with no drawdown)
    are capped at 999.0 so the dict serialises cleanly to JSON.
    """
    def _cap(v: float) -> float:
        return _INF_SENTINEL if math.isinf(v) else v

    return {
        "sharpe_ratio": _cap(sharpe_ratio(returns)),
        "sortino_ratio": _cap(sortino_ratio(returns)),
        "max_drawdown": max_drawdown(returns),
        "cagr": cagr(returns),
        "calmar_ratio": _cap(calmar_ratio(returns)),
    }
