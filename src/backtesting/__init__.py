"""
src/backtesting — Backtesting engine, strategy framework, and performance metrics.
"""
from src.backtesting.engine import BacktestEngine, BacktestResult
from src.backtesting.metrics import compute_all_metrics
from src.backtesting.strategy import BaseStrategy, EnsembleStrategy

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BaseStrategy",
    "EnsembleStrategy",
    "compute_all_metrics",
]
