"""
src/api/routes/portfolio.py — /portfolio route handler.

Runs a per-ticker backtest for each asset in the portfolio, then aggregates
performance metrics as a weighted average across tickers.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.models import PortfolioRequest, PortfolioResponse
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


@router.post("/portfolio", response_model=PortfolioResponse)
async def portfolio(request: PortfolioRequest) -> PortfolioResponse:
    """Compute blended portfolio metrics by running per-ticker backtests.

    For each ticker in *request.tickers*, fetches OHLCV data, builds technical
    features, and runs a neutral-signal
    :class:`~src.backtesting.engine.BacktestEngine`.  The resulting per-ticker
    metrics are combined as a weighted average using *request.weights*.

    Args:
        request: Validated :class:`~src.api.models.PortfolioRequest` payload.

    Returns:
        :class:`~src.api.models.PortfolioResponse` with blended metric values.

    Raises:
        HTTPException: 500 on any unexpected error during data fetch or
            backtest execution.
    """
    try:
        all_metrics: list[dict[str, float]] = []

        for ticker in request.tickers:
            df = fetch_market_data(
                ticker=ticker,
                start=request.start,
                end=request.end,
            )
            df = build_features(df)

            signals = _neutral_signals(len(df))
            strategy = EnsembleStrategy(ensemble_signals=signals)

            engine = BacktestEngine(
                df=df,
                strategy=strategy,
            )
            result = engine.run()
            all_metrics.append(result.metrics)

        # Compute weighted-average blended metrics across all tickers.
        if not all_metrics:
            blended: dict[str, float] = {}
        else:
            metric_keys = list(all_metrics[0].keys())
            blended = {
                key: sum(
                    weight * ticker_metrics[key]
                    for weight, ticker_metrics in zip(request.weights, all_metrics)
                )
                for key in metric_keys
            }

        return PortfolioResponse(
            tickers=request.tickers,
            weights=request.weights,
            metrics=blended,
        )

    except Exception as exc:
        logger.exception("Unhandled error in /portfolio: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
