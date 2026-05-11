"""
src/models/ensemble.py — Ensemble signal combiner for transformer + sentiment signals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EnsembleSignal:
    """Combined signal from the transformer predictor and sentiment scorer."""

    transformer_score: float  # normalized to [-1, 1] via tanh
    sentiment_score: float    # already in [-1, 1]
    ensemble_score: float     # weighted combination
    confidence: float         # abs(ensemble_score)


def generate_ensemble_signal(
    transformer_pred: float,
    sentiment_score: float,
    transformer_weight: float = 0.6,
    sentiment_weight: float = 0.4,
) -> EnsembleSignal:
    """
    Combine a raw transformer prediction with a sentiment score into a single signal.

    Args:
        transformer_pred: Raw scalar prediction from PriceTransformer (any range).
        sentiment_score: Sentiment score already in [-1.0, 1.0].
        transformer_weight: Weight for the transformer component (default 0.6).
        sentiment_weight: Weight for the sentiment component (default 0.4).

    Returns:
        An EnsembleSignal dataclass.

    Raises:
        ValueError: If weights do not sum to 1.0 (within 1e-6 tolerance).
    """
    if abs(transformer_weight + sentiment_weight - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got "
            f"transformer_weight={transformer_weight} + sentiment_weight={sentiment_weight} "
            f"= {transformer_weight + sentiment_weight}"
        )

    norm_transformer = math.tanh(transformer_pred)
    ensemble_score = transformer_weight * norm_transformer + sentiment_weight * sentiment_score
    confidence = abs(ensemble_score)

    return EnsembleSignal(
        transformer_score=norm_transformer,
        sentiment_score=sentiment_score,
        ensemble_score=ensemble_score,
        confidence=confidence,
    )


def batch_ensemble_signals(
    transformer_preds: list[float],
    sentiment_scores: list[float],
    transformer_weight: float = 0.6,
    sentiment_weight: float = 0.4,
) -> list[EnsembleSignal]:
    """
    Generate ensemble signals for a batch of predictions.

    Args:
        transformer_preds: List of raw transformer predictions.
        sentiment_scores: List of sentiment scores in [-1.0, 1.0].
        transformer_weight: Weight for the transformer component.
        sentiment_weight: Weight for the sentiment component.

    Returns:
        List of EnsembleSignal, one per input pair.

    Raises:
        ValueError: If the input lists differ in length, or if weights don't sum to 1.0.
    """
    if len(transformer_preds) != len(sentiment_scores):
        raise ValueError(
            f"transformer_preds and sentiment_scores must have the same length, "
            f"got {len(transformer_preds)} and {len(sentiment_scores)}"
        )

    return [
        generate_ensemble_signal(tp, ss, transformer_weight, sentiment_weight)
        for tp, ss in zip(transformer_preds, sentiment_scores)
    ]
