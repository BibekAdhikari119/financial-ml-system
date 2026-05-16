"""
src/api/models.py — Pydantic v2 request/response schemas for the Financial ML API.
"""
from __future__ import annotations

import datetime

from pydantic import BaseModel, model_validator


class PredictRequest(BaseModel):
    """Request schema for the /predict endpoint."""

    ticker: str
    start: str  # ISO date e.g. "2023-01-01"
    end: str    # ISO date
    window_size: int = 60

    @model_validator(mode="after")
    def end_after_start(self) -> "PredictRequest":
        """Validate that end date is strictly after start date."""
        try:
            start_dt = datetime.date.fromisoformat(self.start)
            end_dt = datetime.date.fromisoformat(self.end)
        except ValueError as exc:
            raise ValueError(f"Invalid date format. Expected YYYY-MM-DD. Details: {exc}") from exc
        if end_dt <= start_dt:
            raise ValueError(
                f"'end' ({self.end}) must be strictly after 'start' ({self.start})."
            )
        return self


class BacktestRequest(BaseModel):
    """Request schema for the /backtest endpoint."""

    ticker: str
    start: str
    end: str
    initial_capital: float = 100_000.0
    transaction_cost_bps: int = 10
    slippage_bps: int = 5
    model_path: str | None = None

    @model_validator(mode="after")
    def end_after_start(self) -> "BacktestRequest":
        """Validate that end date is strictly after start date."""
        try:
            start_dt = datetime.date.fromisoformat(self.start)
            end_dt = datetime.date.fromisoformat(self.end)
        except ValueError as exc:
            raise ValueError(f"Invalid date format. Expected YYYY-MM-DD. Details: {exc}") from exc
        if end_dt <= start_dt:
            raise ValueError(
                f"'end' ({self.end}) must be strictly after 'start' ({self.start})."
            )
        return self


class PortfolioRequest(BaseModel):
    """Request schema for the /portfolio endpoint."""

    tickers: list[str]
    weights: list[float]
    start: str
    end: str

    @model_validator(mode="after")
    def end_after_start(self) -> "PortfolioRequest":
        """Validate that end date is strictly after start date."""
        try:
            start_dt = datetime.date.fromisoformat(self.start)
            end_dt = datetime.date.fromisoformat(self.end)
        except ValueError as exc:
            raise ValueError(f"Invalid date format. Expected YYYY-MM-DD. Details: {exc}") from exc
        if end_dt <= start_dt:
            raise ValueError(
                f"'end' ({self.end}) must be strictly after 'start' ({self.start})."
            )
        return self

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "PortfolioRequest":
        """Validate that weights sum to 1.0 within tolerance and match tickers length."""
        if len(self.tickers) != len(self.weights):
            raise ValueError(
                f"'tickers' and 'weights' must have the same length, "
                f"got {len(self.tickers)} and {len(self.weights)}."
            )
        total = sum(self.weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Portfolio weights must sum to 1.0, got {total:.8f}."
            )
        return self


class PredictResponse(BaseModel):
    """Response schema for the /predict endpoint."""

    ticker: str
    signal: float
    confidence: float
    timestamp: str  # ISO date string of last data point


class BacktestResponse(BaseModel):
    """Response schema for the /backtest endpoint."""

    ticker: str
    metrics: dict[str, float]
    equity_curve: list[float]
    dates: list[str]  # ISO date strings
    benchmark_metrics: dict[str, float] = {}
    benchmark_equity_curve: list[float] = []
    benchmark_dates: list[str] = []


class PortfolioResponse(BaseModel):
    """Response schema for the /portfolio endpoint."""

    tickers: list[str]
    weights: list[float]
    metrics: dict[str, float]


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str
    version: str
