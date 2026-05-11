"""
src/api/routes/predict.py — /predict route handler.

Fetches market data, builds technical features, and returns an ensemble signal.
Model loading from MLflow is a placeholder; a neutral signal is returned when
no model path is configured.
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException

from src.api.models import PredictRequest, PredictResponse
from src.data.market import fetch_market_data
from src.features.technical import build_features
from src.models.ensemble import generate_ensemble_signal

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Generate a trading signal for *ticker* over the requested date range.

    If the ``MLFLOW_MODEL_PATH`` environment variable is set, a warning is
    logged that model loading is not yet implemented and the endpoint falls
    back to a neutral placeholder signal.

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
        mlflow_model_path = os.environ.get("MLFLOW_MODEL_PATH")
        if mlflow_model_path:
            logger.warning(
                "MLFLOW_MODEL_PATH is set ('%s') but model loading is not yet "
                "implemented. Falling back to neutral placeholder signal.",
                mlflow_model_path,
            )

        df = fetch_market_data(
            ticker=request.ticker,
            start=request.start,
            end=request.end,
        )
        df = build_features(df)

        # Placeholder: neutral signal until real model loading is wired in.
        logger.warning(
            "Using placeholder neutral signal (transformer_pred=0.0, "
            "sentiment_score=0.0) for ticker '%s'.",
            request.ticker,
        )
        ensemble_signal = generate_ensemble_signal(
            transformer_pred=0.0,
            sentiment_score=0.0,
        )

        last_date = df.index[-1]
        # Normalise to a plain date string regardless of timestamp precision.
        if hasattr(last_date, "date"):
            timestamp = last_date.date().isoformat()
        else:
            timestamp = str(last_date)[:10]

        return PredictResponse(
            ticker=request.ticker,
            signal=ensemble_signal.ensemble_score,
            confidence=ensemble_signal.confidence,
            timestamp=timestamp,
        )

    except Exception as exc:
        logger.exception(
            "Unhandled error in /predict for ticker '%s': %s",
            request.ticker,
            exc,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from exc
