"""
src/api/_model_loader.py — Single source of truth for loading a trained
PriceTransformer and its fitted StandardScaler from an MLflow artifact path.
"""
import os
import pickle
import logging
from functools import lru_cache

import mlflow.pytorch
from sklearn.preprocessing import StandardScaler

from src.models.transformer import PriceTransformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def load_model_and_scaler(model_path: str) -> tuple[PriceTransformer, StandardScaler]:
    """Load PriceTransformer and its fitted StandardScaler from an MLflow artifact path.

    Args:
        model_path: MLflow model URI, e.g. "runs:/<run_id>/model"

    Returns:
        (model in eval mode, fitted StandardScaler)

    Raises:
        RuntimeError: if model or scaler cannot be loaded
    """
    try:
        model: PriceTransformer = mlflow.pytorch.load_model(model_path)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e

    # Derive run_id from path "runs:/<run_id>/model" or "runs:/<run_id>/..."
    try:
        run_id = model_path.split("/")[1] if model_path.startswith("runs:/") else None
        if run_id is None:
            raise ValueError(f"Cannot parse run_id from model_path: {model_path}")
        scaler_local = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/scaler.pkl")
        with open(scaler_local, "rb") as f:
            scaler: StandardScaler = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler for run {run_id}: {e}") from e

    return model, scaler
