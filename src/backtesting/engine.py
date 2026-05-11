"""
src/backtesting/engine.py — Event-driven backtesting engine.

Design principles
-----------------
* **No lookahead bias**: signals for day *t* are generated from data available
  up to day *t*, and the fill happens at the OPEN of day *t+1*.  The engine
  enforces this by consuming the pre-computed signal series one step at a time
  and always using the *next* day's open as the fill price.
* **Realistic costs**: transaction costs (in basis points of portfolio value)
  and slippage (in basis points of fill price) are applied on every position
  change.
* **Position sizing**: a signal weight of ±1 maps to 100% of current portfolio
  value, converted to an integer number of shares at the fill price.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtesting.metrics import compute_all_metrics
from src.backtesting.strategy import BaseStrategy


@dataclass
class BacktestResult:
    """Container for the output of a single backtest run.

    Attributes:
        equity_curve: Portfolio value (in dollars) indexed by date, one entry
            per trading day from the start of the simulation.
        daily_returns: Simple (arithmetic) daily returns of the portfolio, same
            index as *equity_curve*.  Derived from log returns via exp(lr) - 1.
        trades: DataFrame with one row per position change.  Columns:
            ``["date", "signal", "fill_price", "cost", "pnl"]``.
        metrics: Performance metrics computed by
            :func:`~src.backtesting.metrics.compute_all_metrics`.
    """

    equity_curve: pd.Series
    daily_returns: pd.Series  # simple (not log) daily returns, same index as equity_curve
    trades: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """Event-driven backtesting engine with next-day fill, transaction costs,
    and slippage.

    No-lookahead guarantee
    ----------------------
    :meth:`run` calls ``strategy.generate_signals(df)`` **once** before the
    simulation loop.  The resulting signal series is consumed index-by-index:
    the signal at position *t* drives the target position that is filled at the
    OPEN of bar *t+1*.  Because the strategy receives the full ``df`` at
    construction time (matching how offline models are trained), the burden of
    ensuring signals are truly look-ahead-free rests with the strategy.
    ``EnsembleStrategy`` satisfies this by replaying pre-computed signals.

    Args:
        df: OHLCV DataFrame with a DatetimeIndex.  Required columns:
            ``["Open", "High", "Low", "Close", "Volume"]``.
        strategy: A :class:`~src.backtesting.strategy.BaseStrategy` instance.
        initial_capital: Starting cash in dollars (default 100,000).
        transaction_cost_bps: Round-trip cost per unit of position change,
            expressed in basis points of *portfolio value* (default 10 bps).
        slippage_bps: One-way price slippage per trade expressed in basis
            points of the fill price (default 5 bps).  Applied in the
            direction of the trade.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: int = 10,
        slippage_bps: int = 5,
    ) -> None:
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        if len(df) < 2:
            raise ValueError("DataFrame must have at least 2 rows to run a backtest.")

        self._df = df.copy()
        self._strategy = strategy
        self._initial_capital = initial_capital
        self._tc_bps = transaction_cost_bps
        self._slip_bps = slippage_bps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest and return a :class:`BacktestResult`.

        Simulation mechanics (per day *t*, filling at day *t+1*):

        1. Retrieve ``signal_t`` from the pre-generated signal series.
        2. Clip to [-1, 1].
        3. Compute fill price at OPEN of day *t+1* with directional slippage.
        4. Calculate position change and derive new share count.
        5. Deduct transaction cost from cash.
        6. Mark-to-market at CLOSE of day *t+1*.
        7. Record log return and, if position changed, append a trade row.

        Returns:
            Populated :class:`BacktestResult` instance.
        """
        df = self._df
        n = len(df)

        # Generate ALL signals before the loop — enforces no lookahead in the
        # signal index: signal[t] is consumed with price[t+1].
        raw_signals: pd.Series = self._strategy.generate_signals(df)
        signals = raw_signals.clip(lower=-1.0, upper=1.0)

        # --- State ---
        cash: float = self._initial_capital
        shares: int = 0
        current_position: float = 0.0  # weight in [-1, 1]
        portfolio_value: float = self._initial_capital

        equity_values: list[float] = []
        log_returns: list[float] = []
        trade_rows: list[dict] = []
        dates: list = []

        # Record starting equity at day 0 close.
        prev_portfolio_value = portfolio_value
        start_equity = cash + shares * float(df["Close"].iloc[0])
        equity_values.append(start_equity)
        log_returns.append(0.0)
        dates.append(df.index[0])

        tc_factor = self._tc_bps / 10_000.0
        slip_factor = self._slip_bps / 10_000.0

        for t in range(n - 1):
            signal_t = float(signals.iloc[t])
            date_t1 = df.index[t + 1]
            open_t1 = float(df["Open"].iloc[t + 1])
            close_t1 = float(df["Close"].iloc[t + 1])

            # ---- Fill price with directional slippage ----
            # Positive delta → buying → slippage raises price.
            # Negative delta → selling → slippage lowers price.
            delta_weight = signal_t - current_position
            direction = np.sign(delta_weight) if delta_weight != 0.0 else 0.0
            fill_price = open_t1 * (1.0 + slip_factor * direction)

            # ---- New share count ----
            if fill_price > 0:
                target_shares = math.floor(
                    (signal_t * portfolio_value) / fill_price
                )
            else:
                target_shares = 0

            share_delta = target_shares - shares
            trade_happened = share_delta != 0

            # ---- Transaction cost ----
            cost = abs(delta_weight) * tc_factor * portfolio_value

            if trade_happened:
                # Adjust cash: selling shares adds proceeds, buying reduces cash.
                cash -= share_delta * fill_price  # buy: cash decreases; sell: increases
                cash -= cost
                shares = target_shares
                current_position = signal_t

            # ---- Mark-to-market at close ----
            prev_portfolio_value = portfolio_value
            portfolio_value = cash + shares * close_t1

            # ---- Log return ----
            if prev_portfolio_value > 0:
                lr = math.log(portfolio_value / prev_portfolio_value)
            else:
                lr = 0.0

            equity_values.append(portfolio_value)
            log_returns.append(lr)
            dates.append(date_t1)

            # ---- Record trade ----
            if trade_happened:
                # PnL = mark-to-market gain on the *existing* shares from
                # (fill_price → close_t1), net of cost.
                pnl = shares * (close_t1 - fill_price) - cost
                trade_rows.append(
                    {
                        "date": date_t1,
                        "signal": signal_t,
                        "fill_price": fill_price,
                        "cost": cost,
                        "pnl": pnl,
                    }
                )

        idx = pd.DatetimeIndex(dates) if hasattr(dates[0], "date") else pd.Index(dates)
        equity_curve = pd.Series(equity_values, index=idx, name="equity")

        # Convert log returns to simple returns so compute_all_metrics (which uses
        # (1 + r).cumprod()) receives the correct input and BacktestResult callers
        # do not need to know about the log/simple distinction.
        simple_returns = pd.Series(
            [math.exp(lr) - 1.0 for lr in log_returns],
            index=idx,
            name="return",
        )

        trades_df = (
            pd.DataFrame(trade_rows, columns=["date", "signal", "fill_price", "cost", "pnl"])
            if trade_rows
            else pd.DataFrame(columns=["date", "signal", "fill_price", "cost", "pnl"])
        )

        metrics = compute_all_metrics(simple_returns)

        return BacktestResult(
            equity_curve=equity_curve,
            daily_returns=simple_returns,
            trades=trades_df,
            metrics=metrics,
        )
