"""
src/api/routes/backtest.py — /backtest route handler.

Fetches market data, builds technical features, runs a neutral-signal backtest
via BacktestEngine + EnsembleStrategy, and returns the equity curve with metrics.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.models import BacktestRequest, BacktestResponse
from src.backtesting.engine import BacktestEngine
from src.backtesting.strategy import EnsembleStrategy
from src.data.market import fetch_market_data
from src.features.technical import build_features
from src.models.ensemble import EnsembleSignal

logger = logging.getLogger(__name__)

router = APIRouter()


def _neutral_signals(n: int) -> list[EnsembleSignal]:
    """Return a list of *n* all-zero :class:`~src.models.ensemble.EnsembleSignal` objects.

    Args:
        n: Number of signals to generate.

    Returns:
        List of neutral EnsembleSignal instances (all fields 0.0).
    """
    return [
        EnsembleSignal(
            transformer_score=0.0,
            sentiment_score=0.0,
            ensemble_score=0.0,
            confidence=0.0,
        )
        for _ in range(n)
    ]


@router.post("/backtest", response_model=BacktestResponse)
async def backtest(request: BacktestRequest) -> BacktestResponse:
    """Run a backtest for *ticker* with a neutral placeholder strategy.

    Fetches OHLCV data, computes technical features, then executes
    :class:`~src.backtesting.engine.BacktestEngine` with an
    :class:`~src.backtesting.strategy.EnsembleStrategy` driven by all-zero
    signals.

    Args:
        request: Validated :class:`~src.api.models.BacktestRequest` payload.

    Returns:
        :class:`~src.api.models.BacktestResponse` containing performance
        metrics, equity curve values, and corresponding date strings.

    Raises:
        HTTPException: 500 on any unexpected error.
    """
    try:
        df = fetch_market_data(
            ticker=request.ticker,
            start=request.start,
            end=request.end,
        )
        df = build_features(df)

        signals = _neutral_signals(len(df))
        strategy = EnsembleStrategy(ensemble_signals=signals)

        engine = BacktestEngine(
            df=df,
            strategy=strategy,
            initial_capital=request.initial_capital,
            transaction_cost_bps=request.transaction_cost_bps,
            slippage_bps=request.slippage_bps,
        )
        result = engine.run()

        equity_curve: list[float] = [float(v) for v in result.equity_curve.values]
        dates: list[str] = [
            idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
            for idx in result.equity_curve.index
        ]

        return BacktestResponse(
            ticker=request.ticker,
            metrics=result.metrics,
            equity_curve=equity_curve,
            dates=dates,
        )

    except Exception as exc:
        logger.exception(
            "Unhandled error in /backtest for ticker '%s': %s",
            request.ticker,
            exc,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from exc
