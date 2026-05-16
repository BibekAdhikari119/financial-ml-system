#!/usr/bin/env python3
"""
scripts/evaluate.py — Evaluate a trained PriceTransformer run vs. buy-and-hold.

Usage:
    python scripts/evaluate.py --run-id <run_id> --ticker AAPL \
        --start 2024-01-01 --end 2025-01-01
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure project root is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtesting.engine import BacktestEngine
from src.backtesting.strategy import BaseStrategy, EnsembleStrategy
from src.data.market import fetch_market_data
from src.features.dataset import TimeSeriesDataset
from src.features.technical import build_features
from src.models.ensemble import EnsembleSignal, batch_ensemble_signals


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark strategy
# ---------------------------------------------------------------------------

class BuyAndHoldStrategy(BaseStrategy):
    """Always fully long — 100% of capital invested from day 1."""

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index, dtype=float, name="signal")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PriceTransformer MLflow run against buy-and-hold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-id", type=str, required=True, dest="run_id",
                        help="MLflow run ID from a completed train.py run.")
    parser.add_argument("--ticker", type=str, required=True,
                        help="Equity ticker symbol for evaluation period.")
    parser.add_argument("--start", type=str, required=True,
                        help="Evaluation start date (ISO format).")
    parser.add_argument("--end", type=str, required=True,
                        help="Evaluation end date (ISO format).")
    parser.add_argument("--initial-capital", type=float, default=100_000.0,
                        dest="initial_capital",
                        help="Starting portfolio value in dollars.")
    parser.add_argument("--transaction-cost-bps", type=int, default=10,
                        dest="transaction_cost_bps",
                        help="Round-trip transaction cost in basis points.")
    parser.add_argument("--slippage-bps", type=int, default=5, dest="slippage_bps",
                        help="One-way slippage in basis points.")
    parser.add_argument("--window-size", type=int, default=60, dest="window_size",
                        help="Fallback window size if not stored in run params.")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Fallback horizon if not stored in run params.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_pct(value: float) -> str:
    """Format a float as a percentage string (e.g. 0.1234 -> '12.34%')."""
    return f"{value * 100:.2f}%"


def _outperforms(metric: str, strategy_val: float, bh_val: float) -> str:
    """Return a check mark when the strategy metric is better than buy-and-hold."""
    # For max_drawdown: less negative (closer to 0) is better.
    if metric == "max_drawdown":
        return "+" if strategy_val > bh_val else ""
    return "+" if strategy_val > bh_val else ""


def _print_comparison(
    ticker: str,
    start: str,
    end: str,
    run_id: str,
    strat_metrics: dict[str, float],
    bh_metrics: dict[str, float],
) -> None:
    """Print a formatted side-by-side comparison table."""
    pct_metrics = {"cagr", "max_drawdown"}
    rows: list[tuple[str, str, str, str]] = []

    metric_order = ["sharpe_ratio", "sortino_ratio", "cagr", "max_drawdown", "calmar_ratio"]
    for metric in metric_order:
        sv = strat_metrics.get(metric, float("nan"))
        bv = bh_metrics.get(metric, float("nan"))
        if metric in pct_metrics:
            sv_str = _format_pct(sv)
            bv_str = _format_pct(bv)
        else:
            sv_str = f"{sv:.4f}"
            bv_str = f"{bv:.4f}"
        beats = _outperforms(metric, sv, bv)
        rows.append((metric, sv_str, bv_str, beats))

    sep = "=" * 68
    header_line = f"{'Metric':<22}{'Strategy':>12}{'Buy & Hold':>12}{'Outperforms':>12}"
    divider = f"{'-' * 22}{'-' * 12}{'-' * 12}{'-' * 12}"

    print(f"\n{sep}")
    print(f"Evaluation: {ticker} ({start} -> {end})  run_id: {run_id[:8]}...")
    print(sep)
    print(header_line)
    print(divider)
    for metric, sv_str, bv_str, beats in rows:
        print(f"{metric:<22}{sv_str:>12}{bv_str:>12}{beats:>12}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    run_id: str = args.run_id
    ticker: str = args.ticker
    start: str = args.start
    end: str = args.end
    initial_capital: float = args.initial_capital
    tc_bps: int = args.transaction_cost_bps
    slip_bps: int = args.slippage_bps
    window_size: int = args.window_size
    horizon: int = args.horizon
    seed: int = args.seed

    # ------------------------------------------------------------------
    # 1. Set seeds.
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ------------------------------------------------------------------
    # 2. Load run parameters logged by train.py.
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "./mlruns"))

    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params

    feature_cols: list[str] = params["feature_cols"].split(",")
    window_size_trained: int = int(params.get("window_size", str(window_size)))
    horizon_trained: int = int(params.get("horizon", str(horizon)))

    print(f"Loaded run {run_id[:8]}...")
    print(f"  Trained features ({len(feature_cols)}): {feature_cols}")
    print(f"  window_size={window_size_trained}, horizon={horizon_trained}")

    # ------------------------------------------------------------------
    # 3. Load model and scaler artefacts.
    # ------------------------------------------------------------------
    print("Loading model from MLflow...")
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    model.eval()

    print("Downloading scaler artefact...")
    # The scaler was logged at the root artefact path, so we download the
    # root directory and locate the .pkl file within it.
    local_artifact_dir: str = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/scaler.pkl"
    )
    with open(local_artifact_dir, "rb") as f:
        scaler = pickle.load(f)

    # ------------------------------------------------------------------
    # 4. Fetch and featurize evaluation data.
    # ------------------------------------------------------------------
    print(f"Fetching evaluation data: {ticker} ({start} -> {end})")
    raw_df = fetch_market_data(ticker, start, end)
    feat_df = build_features(raw_df)

    # Validate that the feature dimension matches what the model expects.
    expected_input_dim: int = model.input_proj.in_features
    if len(feature_cols) != expected_input_dim:
        raise ValueError(
            f"Feature count mismatch: model expects {expected_input_dim} features "
            f"but the training run logged {len(feature_cols)} feature columns."
        )

    missing_cols = [c for c in feature_cols if c not in feat_df.columns]
    if missing_cols:
        raise ValueError(
            f"Evaluation DataFrame is missing feature columns: {missing_cols}"
        )

    # ------------------------------------------------------------------
    # 5. Build evaluation Dataset — reuse training scaler (no re-fitting).
    # ------------------------------------------------------------------
    eval_ds = TimeSeriesDataset(
        feat_df, feature_cols, "Close",
        window_size=window_size_trained,
        horizon=horizon_trained,
        scaler=scaler,
        fit_scaler=False,
    )
    eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False)

    print(f"Evaluation samples: {len(eval_ds)}")

    # ------------------------------------------------------------------
    # 6. Run inference.
    # ------------------------------------------------------------------
    preds: list[float] = []
    with torch.no_grad():
        for X_batch, _ in eval_loader:
            out = model(X_batch)
            preds.extend(out.squeeze(-1).tolist())

    # ------------------------------------------------------------------
    # 7. Build signal list — pad first window_size entries with neutral.
    # ------------------------------------------------------------------
    neutral = EnsembleSignal(0.0, 0.0, 0.0, 0.0)
    pad: list[EnsembleSignal] = [neutral] * window_size_trained
    model_signals: list[EnsembleSignal] = batch_ensemble_signals(
        preds, [0.0] * len(preds)
    )
    signals: list[EnsembleSignal] = pad + model_signals

    # Trim to match feat_df exactly (handles edge-case where dataset has fewer
    # samples than window_size + len(preds) due to horizon offset).
    if len(signals) > len(feat_df):
        signals = signals[: len(feat_df)]
    elif len(signals) < len(feat_df):
        deficit = len(feat_df) - len(signals)
        signals = signals + [neutral] * deficit

    assert len(signals) == len(feat_df), (
        f"Signal length {len(signals)} != feat_df length {len(feat_df)}"
    )

    # ------------------------------------------------------------------
    # 8. Run ensemble strategy backtest.
    # ------------------------------------------------------------------
    strategy = EnsembleStrategy(signals)
    engine = BacktestEngine(
        feat_df, strategy,
        initial_capital=initial_capital,
        transaction_cost_bps=tc_bps,
        slippage_bps=slip_bps,
    )
    strat_result = engine.run()

    # ------------------------------------------------------------------
    # 9. Buy-and-hold benchmark.
    # ------------------------------------------------------------------
    bh_engine = BacktestEngine(
        feat_df, BuyAndHoldStrategy(),
        initial_capital=initial_capital,
        transaction_cost_bps=tc_bps,
        slippage_bps=slip_bps,
    )
    bh_result = bh_engine.run()

    # ------------------------------------------------------------------
    # 10. Print comparison table.
    # ------------------------------------------------------------------
    _print_comparison(ticker, start, end, run_id, strat_result.metrics, bh_result.metrics)

    # ------------------------------------------------------------------
    # 11. Log evaluation metrics back to the original MLflow run.
    # ------------------------------------------------------------------
    for k, v in strat_result.metrics.items():
        client.log_metric(run_id, f"eval_strategy_{k}", v)
    for k, v in bh_result.metrics.items():
        client.log_metric(run_id, f"eval_bh_{k}", v)

    print(f"Logged evaluation metrics back to run {run_id[:8]}...")

    # ------------------------------------------------------------------
    # 12. Save comparison to CSV and upload as artefact on the run.
    # ------------------------------------------------------------------
    comparison_rows: list[dict[str, object]] = []
    for metric in ["sharpe_ratio", "sortino_ratio", "cagr", "max_drawdown", "calmar_ratio"]:
        sv = strat_result.metrics.get(metric, float("nan"))
        bv = bh_result.metrics.get(metric, float("nan"))
        comparison_rows.append({
            "metric": metric,
            "strategy": sv,
            "buy_and_hold": bv,
            "outperforms": sv > bv if metric != "max_drawdown" else sv > bv,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w"
    ) as tf:
        comparison_df.to_csv(tf, index=False)
        csv_path = tf.name

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(csv_path, artifact_path="evaluation")
    os.unlink(csv_path)

    print(f"Saved comparison CSV to MLflow run {run_id[:8]}... (evaluation/)")


if __name__ == "__main__":
    main()
