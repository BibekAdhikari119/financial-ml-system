"""
src/api/app.py — FastAPI application entry point for the Financial ML API.

Environment variables
---------------------
ALLOWED_ORIGINS : comma-separated list of allowed CORS origins (default "*").
PORT            : TCP port to listen on when run directly (default 8000).
"""
from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import HealthResponse
from src.api.routes import backtest_router, portfolio_router, predict_router

load_dotenv()

app = FastAPI(
    title="Financial ML API",
    version="1.0.0",
    description=(
        "Production-grade REST API for a financial ML trading system. "
        "Provides signal prediction, backtesting, and portfolio analytics."
    ),
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
_allowed_origins_raw: str = os.environ.get("ALLOWED_ORIGINS", "*")
_allowed_origins: list[str] = [o.strip() for o in _allowed_origins_raw.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
_API_PREFIX = "/api/v1"

app.include_router(predict_router, prefix=_API_PREFIX, tags=["predict"])
app.include_router(backtest_router, prefix=_API_PREFIX, tags=["backtest"])
app.include_router(portfolio_router, prefix=_API_PREFIX, tags=["portfolio"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Return the service health status and current API version.

    Returns:
        :class:`~src.api.models.HealthResponse` with ``status="ok"`` and
        the current semantic version string.
    """
    return HealthResponse(status="ok", version="1.0.0")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=_port)
