"""
src/models/rl_agent.py — Stable-Baselines3 PPO trading agent with custom Gym environment.

Uses `gymnasium` (not legacy `gym`) — required for SB3 >= 2.3.
"""
from __future__ import annotations

import math
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces
from stable_baselines3 import PPO


class TradingEnv(gymnasium.Env):
    """
    A continuous-action trading environment backed by a feature DataFrame.

    Observation:  [feature_cols..., position_weight, unrealized_pnl_ratio, cash_ratio]
    Action:       target position weight in [-max_position, max_position]
    Reward:       log return of portfolio value, clipped to [-1, 1]

    Execution model (no lookahead)
    ------------------------------
    The agent observes close_t, decides a target weight, and the order is filled
    at the OPEN of bar t+1 (next_open).  Mark-to-market uses close_t+1.
    Actual shares are tracked so cash and equity are never double-counted.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        initial_capital: float = 100_000.0,
        transaction_cost_bps: int = 10,
        slippage_bps: int = 5,
        max_position: float = 1.0,
    ) -> None:
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.max_position = max_position

        obs_dim = len(feature_cols) + 3  # features + position_weight + unreal_pnl_ratio + cash_ratio
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # State variables — initialised properly in reset()
        self.current_step: int = 0
        self.position_weight: float = 0.0  # current target weight in [-max_position, max_position]
        self.shares: float = 0.0           # actual shares held (fractional allowed)
        self.cash: float = initial_capital
        self.portfolio_value: float = initial_capital

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        features = row[self.feature_cols].values.astype(np.float32)

        unrealized_pnl_ratio = (
            (self.portfolio_value - self.initial_capital) / self.initial_capital
            if self.initial_capital != 0
            else 0.0
        )
        cash_ratio = self.cash / self.portfolio_value if self.portfolio_value > 0 else 1.0

        obs = np.concatenate(
            [features, [self.position_weight, unrealized_pnl_ratio, cash_ratio]]
        ).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.position_weight = 0.0
        self.shares = 0.0
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target_weight = float(np.clip(action[0], -self.max_position, self.max_position))

        # Observed price at current step (no lookahead)
        current_close = float(self.df["Close"].iloc[self.current_step])

        # Execution price: OPEN of next bar (avoids lookahead on the close)
        if self.current_step + 1 < len(self.df):
            next_open = float(self.df["Open"].iloc[self.current_step + 1])
        else:
            next_open = current_close  # terminal step fallback

        # Use weight direction to determine slippage sign before shares are known.
        # This breaks the circularity (fill_price ↔ target_shares) with one estimation
        # step, ensuring shares are always booked at the same price as cash is debited.
        weight_direction = float(np.sign(target_weight - self.position_weight))
        fill_price = (
            next_open * (1.0 + self.slippage_bps / 10_000.0 * weight_direction)
            if next_open != 0
            else 1.0
        )

        # Desired shares at the execution (fill) price — consistent with cash debit below
        target_shares = (target_weight * self.portfolio_value) / fill_price if fill_price != 0 else 0.0
        share_delta = target_shares - self.shares

        # Transaction cost on the dollar value of the trade
        cost = abs(share_delta) * fill_price * self.transaction_cost_bps / 10_000.0

        old_portfolio_value = self.portfolio_value

        # Update cash: buying reduces cash, selling increases it; costs always reduce cash
        self.cash -= share_delta * fill_price + cost
        self.shares = target_shares
        self.position_weight = target_weight

        # Advance to next bar and mark-to-market at its close
        self.current_step += 1
        new_close = float(self.df["Close"].iloc[self.current_step])

        new_portfolio_value = self.cash + self.shares * new_close
        # Guard against log(0) or negative portfolio value
        new_portfolio_value = max(new_portfolio_value, 1e-8)

        # Log return clipped to [-1, 1] to keep gradients stable
        reward = float(
            np.clip(
                np.log(new_portfolio_value / max(old_portfolio_value, 1e-8)),
                -1.0,
                1.0,
            )
        )

        self.portfolio_value = new_portfolio_value

        terminated = self.current_step >= len(self.df) - 1
        obs = self._get_obs()

        info: dict[str, Any] = {
            "portfolio_value": self.portfolio_value,
            "position": self.position_weight,
            "capital": self.cash,
        }
        return obs, reward, terminated, False, info

    def render(self) -> None:  # type: ignore[override]
        pass  # noqa: unnecessary-pass


def train_rl_agent(
    env: TradingEnv,
    total_timesteps: int = 100_000,
    seed: int = 42,
    mlflow_run_id: str | None = None,
) -> PPO:
    """
    Train a PPO agent on the given TradingEnv.

    Args:
        env: A TradingEnv (or compatible) instance.
        total_timesteps: Total environment steps for training.
        seed: Random seed for the PPO algorithm.
        mlflow_run_id: If provided, logs hyperparameters to MLflow.

    Returns:
        Trained PPO model.
    """
    model = PPO("MlpPolicy", env, seed=seed, verbose=0)
    model.learn(total_timesteps=total_timesteps)

    if mlflow_run_id is not None:
        client = mlflow.MlflowClient()
        client.log_param(mlflow_run_id, "total_timesteps", str(total_timesteps))
        client.log_param(mlflow_run_id, "seed", str(seed))

    return model
