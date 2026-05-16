"""
src/api/routes/backtest.py — /backtest route handler.

Fetches market data, builds technical features, runs a strategy backtest
via BacktestEngine + EnsembleStrategy (optionally driven by a trained
PriceTransformer), and returns the equity curve with metrics alongside a
buy-and-hold benchmark.
"""
from __future__ import annotations

import logging
import os

import torch
from fastapi import APIRouter, HTTPException
from torch.utils.data import DataLoader

from src.api._model_loader import load_model_and_scaler
from src.api.models import BacktestRequest, BacktestResponse
from src.backtesting.engine import BacktestEngine
from src.backtesting.strategy import EnsembleStrategy
from src.data.market import fetch_market_data
from src.features.dataset import TimeSeriesDataset
from src.features.technical import build_features
from src.models.ensemble import EnsembleSignal, batch_ensemble_signals

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
    """Run a backtest for *ticker*, optionally using a trained PriceTransformer.

    Fetches OHLCV data, computes technical features, then executes
    :class:`~src.backtesting.engine.BacktestEngine` with an
    :class:`~src.backtesting.strategy.EnsembleStrategy`.  When a model path is
    available (via ``request.model_path`` or ``MLFLOW_MODEL_PATH`` env var),
    real transformer predictions drive the signals; otherwise neutral signals
    are used.

    A buy-and-hold benchmark is always computed alongside the strategy.

    Args:
        request: Validated :class:`~src.api.models.BacktestRequest` payload.

    Returns:
        :class:`~src.api.models.BacktestResponse` containing performance
        metrics, equity curve values, corresponding date strings, and the
        equivalent benchmark fields.

    Raises:
        HTTPException: 500 on any unexpected error.
    """
    try:
        df = fetch_market_data(
            ticker=request.ticker,
            start=request.start,
            end=request.end,
        )
        feat_df = build_features(df)

        n = len(feat_df)

        # --- Strategy signals ---
        model_path = request.model_path or os.environ.get("MLFLOW_MODEL_PATH")
        window_size = int(os.environ.get("MLFLOW_WINDOW_SIZE", "60"))

        if model_path:
            try:
                model, scaler = load_model_and_scaler(model_path)
                feature_cols = [c for c in feat_df.columns
                                if c not in {"Open", "High", "Low", "Close", "Volume"}]
                ds = TimeSeriesDataset(
                    feat_df, feature_cols, "Close",
                    window_size=window_size, horizon=1,
                    scaler=scaler, fit_scaler=False,
                )
                preds: list[float] = []
                with torch.no_grad():
                    loader = DataLoader(ds, batch_size=64, shuffle=False)
                    for X_batch, _ in loader:
                        preds.extend(model(X_batch).squeeze(-1).tolist())
                pad = [EnsembleSignal(0.0, 0.0, 0.0, 0.0)] * window_size
                model_sigs = batch_ensemble_signals(preds, [0.0] * len(preds))
                signals = (pad + model_sigs)[:n]
            except Exception as e:
                logger.warning("Model loading failed, using neutral signals: %s", e)
                signals = _neutral_signals(n)
        else:
            signals = _neutral_signals(n)

        # --- Strategy backtest ---
        strategy = EnsembleStrategy(ensemble_signals=signals)
        engine = BacktestEngine(
            df=feat_df,
            strategy=strategy,
            initial_capital=request.initial_capital,
            transaction_cost_bps=request.transaction_cost_bps,
            slippage_bps=request.slippage_bps,
        )
        result = engine.run()

        # --- Buy-and-hold benchmark (always run) ---
        bh_signals = [EnsembleSignal(1.0, 0.0, 1.0, 1.0)] * n
        bh_strategy = EnsembleStrategy(ensemble_signals=bh_signals)
        bh_engine = BacktestEngine(
            df=feat_df,
            strategy=bh_strategy,
            initial_capital=request.initial_capital,
            transaction_cost_bps=request.transaction_cost_bps,
            slippage_bps=request.slippage_bps,
        )
        bh_result = bh_engine.run()

        return BacktestResponse(
            ticker=request.ticker,
            metrics=result.metrics,
            equity_curve=[float(v) for v in result.equity_curve.values],
            dates=[
                idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
                for idx in result.equity_curve.index
            ],
            benchmark_metrics=bh_result.metrics,
            benchmark_equity_curve=[float(v) for v in bh_result.equity_curve.values],
            benchmark_dates=[
                idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
                for idx in bh_result.equity_curve.index
            ],
        )

    except Exception as exc:
        logger.exception(
            "Unhandled error in /backtest for ticker '%s': %s",
            request.ticker,
            exc,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from exc
