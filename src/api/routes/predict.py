"""
src/api/routes/predict.py — /predict route handler.

Fetches market data, builds technical features, and returns an ensemble signal.
When MLFLOW_MODEL_PATH is set, real inference is performed via PriceTransformer;
otherwise a neutral signal is returned.
"""
from __future__ import annotations

import logging
import os

import torch
from fastapi import APIRouter, HTTPException

from src.api._model_loader import load_model_and_scaler
from src.api.models import PredictRequest, PredictResponse
from src.data.market import fetch_market_data
from src.features.dataset import TimeSeriesDataset
from src.features.technical import build_features
from src.models.ensemble import EnsembleSignal, generate_ensemble_signal

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Generate a trading signal for *ticker* over the requested date range.

    When ``MLFLOW_MODEL_PATH`` is set the most recent window is fed through the
    trained PriceTransformer to produce a real prediction.  If the path is
    unset, or if inference fails, a neutral placeholder signal is returned.

    Args:
        request: Validated :class:`~src.api.models.PredictRequest` payload.

    Returns:
        :class:`~src.api.models.PredictResponse` with signal, confidence, and
        the ISO date of the last available data point.

    Raises:
        HTTPException: 500 on any unexpected error during data fetch or signal
            generation.
    """
    try:
        df = fetch_market_data(
            ticker=request.ticker,
            start=request.start,
            end=request.end,
        )
        df = build_features(df)

        model_path = os.environ.get("MLFLOW_MODEL_PATH")
        window_size = int(os.environ.get("MLFLOW_WINDOW_SIZE", "60"))

        signal_obj: EnsembleSignal
        if model_path:
            try:
                model, scaler = load_model_and_scaler(model_path)
                feature_cols = [c for c in df.columns if c not in {"Open", "High", "Low", "Close", "Volume"}]
                ds = TimeSeriesDataset(
                    df, feature_cols, "Close",
                    window_size=window_size, horizon=1,
                    scaler=scaler, fit_scaler=False,
                )
                if len(ds) == 0:
                    raise ValueError("Not enough data for one full window")
                # Use only the last window
                X_last, _ = ds[len(ds) - 1]
                with torch.no_grad():
                    pred = float(model(X_last.unsqueeze(0)).squeeze())
                signal_obj = generate_ensemble_signal(transformer_pred=pred, sentiment_score=0.0)
            except Exception as e:
                logger.warning("Model inference failed, using neutral signal: %s", e)
                signal_obj = EnsembleSignal(0.0, 0.0, 0.0, 0.0)
        else:
            logger.info("MLFLOW_MODEL_PATH not set, returning neutral signal")
            signal_obj = EnsembleSignal(0.0, 0.0, 0.0, 0.0)

        last_date = df.index[-1]
        # Normalise to a plain date string regardless of timestamp precision.
        if hasattr(last_date, "date"):
            timestamp = last_date.date().isoformat()
        else:
            timestamp = str(last_date)[:10]

        return PredictResponse(
            ticker=request.ticker,
            signal=signal_obj.ensemble_score,
            confidence=signal_obj.confidence,
            timestamp=timestamp,
        )

    except Exception as exc:
        logger.exception(
            "Unhandled error in /predict for ticker '%s': %s",
            request.ticker,
            exc,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from exc
