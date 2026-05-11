"""
tests/models/test_rl_agent.py — Unit tests for TradingEnv and train_rl_agent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO

from src.models.rl_agent import TradingEnv, train_rl_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_COLS = ["rsi_14", "macd_line", "sma_20"]


def _make_df(n_rows: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic OHLCV + feature DataFrame deterministically.

    The Close column is a random walk so that price dynamics are realistic
    enough for the environment to function without division-by-zero issues.
    """
    rng = np.random.default_rng(seed)

    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n_rows))
    close = np.maximum(close, 1.0)  # keep prices positive

    df = pd.DataFrame(
        {
            "Open": close * rng.uniform(0.99, 1.01, n_rows),
            "High": close * rng.uniform(1.00, 1.02, n_rows),
            "Low": close * rng.uniform(0.98, 1.00, n_rows),
            "Close": close,
            "Volume": rng.integers(1_000, 10_000, n_rows).astype(float),
            # Synthetic technical features
            "rsi_14": rng.uniform(20, 80, n_rows),
            "macd_line": rng.normal(0, 1, n_rows),
            "sma_20": close * rng.uniform(0.97, 1.03, n_rows),
        }
    )
    return df


def _make_env(n_rows: int = 50, seed: int = 42) -> TradingEnv:
    df = _make_df(n_rows=n_rows, seed=seed)
    return TradingEnv(df=df, feature_cols=FEATURE_COLS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_check_env() -> None:
    """gymnasium env_checker must pass with no errors."""
    from gymnasium.utils.env_checker import check_env

    env = _make_env()
    # check_env raises AssertionError or Exception on failure
    check_env(env, warn=True)


def test_transaction_cost_deducted() -> None:
    """Portfolio value after a non-zero position change must reflect transaction cost."""
    env = _make_env()
    env.reset(seed=0)

    initial_portfolio = env.portfolio_value

    # Force a large position change (from 0 → 1.0)
    action = np.array([1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    # If there were no cost, portfolio change would only be price-driven.
    # With transaction cost + slippage, the value consumed by costs must be > 0.
    # We verify portfolio_value is strictly less than initial_capital + unrealised gain.
    # The simplest check: transaction cost > 0 means info["capital"] < initial_capital.
    assert info["capital"] < initial_portfolio, (
        "Expected capital to decrease due to transaction cost, "
        f"got capital={info['capital']} >= initial={initial_portfolio}"
    )


def test_deterministic_reset() -> None:
    """reset(seed=42) called twice on the same env yields identical first observations."""
    env = _make_env()

    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)

    np.testing.assert_array_equal(
        obs1, obs2,
        err_msg="reset(seed=42) produced different observations across two calls"
    )


def test_train_rl_agent_short_run() -> None:
    """train_rl_agent completes without error and returns a PPO instance."""
    env = _make_env(n_rows=100)
    model = train_rl_agent(env, total_timesteps=500, seed=0)

    assert isinstance(model, PPO), (
        f"Expected PPO instance, got {type(model)}"
    )
