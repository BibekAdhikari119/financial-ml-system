"""
src/api/routes/__init__.py — Router registry for the Financial ML API.
"""
from src.api.routes.backtest import router as backtest_router
from src.api.routes.portfolio import router as portfolio_router
from src.api.routes.predict import router as predict_router

__all__ = ["predict_router", "backtest_router", "portfolio_router"]
