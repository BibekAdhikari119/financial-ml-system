"""
src/backtesting/strategy.py — Abstract strategy base class and EnsembleStrategy implementation.

Strategies return a pd.Series of target position weights in [-1, 1], indexed by
the DatetimeIndex of the price DataFrame passed in.  A weight of +1 means 100%
long, -1 means 100% short, 0 means flat.
"""
from __future__ import annotations

import abc

import pandas as pd

from src.models.ensemble import EnsembleSignal


class BaseStrategy(abc.ABC):
    """Abstract base class for all backtesting strategies.

    Subclasses must implement :meth:`generate_signals`.  Signals must be
    indexed by ``df``'s DatetimeIndex and must not use any forward-looking
    information (no lookahead bias).
    """

    @abc.abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return target position weights in [-1, 1] for each row in *df*.

        Args:
            df: OHLCV DataFrame with a DatetimeIndex.  At minimum it must
                contain columns ``["Open", "High", "Low", "Close", "Volume"]``.

        Returns:
            pd.Series of float weights in [-1, 1], length == len(df),
            indexed by ``df.index``.
        """
        ...


class EnsembleStrategy(BaseStrategy):
    """Strategy that replays pre-computed :class:`~src.models.ensemble.EnsembleSignal` objects.

    The ensemble scores were generated offline from historical data (no
    lookahead), so consuming them sequentially in the engine is safe.

    If ``len(ensemble_signals) != len(df)`` the signal list is either
    truncated (if longer) or zero-padded at the end (if shorter) to match
    the DataFrame length.

    Args:
        ensemble_signals: Ordered list of EnsembleSignal, one per trading day.
    """

    def __init__(self, ensemble_signals: list[EnsembleSignal]) -> None:
        self._signals = ensemble_signals

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Map pre-computed ensemble scores onto *df*'s index.

        Args:
            df: OHLCV DataFrame whose index is used to label the output.

        Returns:
            pd.Series of ``ensemble_score`` values aligned to ``df.index``.
            Values outside the range [-1, 1] are possible but will be clipped
            by :class:`~src.backtesting.engine.BacktestEngine` before use.
        """
        n = len(df)
        signals = self._signals

        if len(signals) > n:
            scores = [s.ensemble_score for s in signals[:n]]
        else:
            scores = [s.ensemble_score for s in signals]
            scores += [0.0] * (n - len(scores))

        return pd.Series(scores, index=df.index, dtype=float, name="signal")
