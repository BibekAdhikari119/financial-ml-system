#!/usr/bin/env python3
"""
scripts/train.py — Full training pipeline CLI for PriceTransformer.

Usage:
    python scripts/train.py --ticker AAPL --start 2020-01-01 --end 2024-01-01
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
import torch
from torch.utils.data import DataLoader

# Ensure project root is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.market import fetch_market_data
from src.features.dataset import TimeSeriesDataset, create_train_val_test_splits
from src.features.technical import build_features
from src.models.transformer import PriceTransformer, train_transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PriceTransformer on market data and log artefacts to MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", type=str, required=True, help="Equity ticker symbol.")
    parser.add_argument("--start", type=str, required=True, help="Start date (ISO format).")
    parser.add_argument("--end", type=str, required=True, help="End date (ISO format).")
    parser.add_argument("--window-size", type=int, default=60, dest="window_size",
                        help="Sliding window size (timesteps).")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Prediction horizon (timesteps ahead).")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate.")
    parser.add_argument("--d-model", type=int, default=128, dest="d_model",
                        help="Transformer d_model dimension.")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads.")
    parser.add_argument("--num-layers", type=int, default=3, dest="num_layers",
                        help="Number of transformer encoder layers.")
    parser.add_argument("--dim-feedforward", type=int, default=256, dest="dim_feedforward",
                        help="Feedforward dimension inside the encoder layer.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability.")
    parser.add_argument("--batch-size", type=int, default=64, dest="batch_size",
                        help="Mini-batch size.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--mlflow-experiment", type=str, default="price-transformer",
                        dest="mlflow_experiment",
                        help="MLflow experiment name.")
    parser.add_argument("--model-name", type=str, default="price-transformer",
                        dest="model_name",
                        help="Registered model name in MLflow Model Registry.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device ('cpu' or 'cuda').")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ticker: str = args.ticker
    start: str = args.start
    end: str = args.end
    window_size: int = args.window_size
    horizon: int = args.horizon
    epochs: int = args.epochs
    lr: float = args.lr
    d_model: int = args.d_model
    nhead: int = args.nhead
    num_layers: int = args.num_layers
    dim_feedforward: int = args.dim_feedforward
    dropout: float = args.dropout
    batch_size: int = args.batch_size
    seed: int = args.seed
    mlflow_experiment: str = args.mlflow_experiment
    model_name: str = args.model_name
    device: str = args.device

    # ------------------------------------------------------------------
    # 1. Set seeds for reproducibility.
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ------------------------------------------------------------------
    # 2. Fetch market data and build features.
    # ------------------------------------------------------------------
    print(f"Fetching market data: {ticker} ({start} -> {end})")
    raw_df = fetch_market_data(ticker, start, end)
    feat_df = build_features(raw_df)

    ohlcv_cols = {"Open", "High", "Low", "Close", "Volume"}
    feature_cols: list[str] = [c for c in feat_df.columns if c not in ohlcv_cols]

    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Total rows after NaN drop: {len(feat_df)}")

    # ------------------------------------------------------------------
    # 3. Strictly time-ordered train/val/test split.
    # ------------------------------------------------------------------
    train_df, val_df, test_df = create_train_val_test_splits(feat_df)

    # ------------------------------------------------------------------
    # 4. Build PyTorch Datasets — scaler fitted on train only.
    # ------------------------------------------------------------------
    train_ds = TimeSeriesDataset(
        train_df, feature_cols, "Close",
        window_size=window_size, horizon=horizon,
        fit_scaler=True,
    )
    val_ds = TimeSeriesDataset(
        val_df, feature_cols, "Close",
        window_size=window_size, horizon=horizon,
        scaler=train_ds.scaler, fit_scaler=False,
    )
    test_ds = TimeSeriesDataset(
        test_df, feature_cols, "Close",
        window_size=window_size, horizon=horizon,
        scaler=train_ds.scaler, fit_scaler=False,
    )

    # ------------------------------------------------------------------
    # 5. DataLoaders — shuffle=False to preserve temporal order.
    # ------------------------------------------------------------------
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # 6. Build the model.
    # ------------------------------------------------------------------
    model = PriceTransformer(
        input_dim=len(feature_cols),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # 7. MLflow run: log params, train, log metrics, save artefacts.
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=f"{ticker}-{start}-{end}") as run:
        run_id: str = run.info.run_id

        # Log all hyperparameters.
        mlflow.log_params({
            "ticker": ticker,
            "start": start,
            "end": end,
            "window_size": window_size,
            "horizon": horizon,
            "epochs": epochs,
            "lr": lr,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "batch_size": batch_size,
            "seed": seed,
            "n_features": len(feature_cols),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "feature_cols": ",".join(feature_cols),  # reconstructed in evaluate.py
        })

        # Train model.
        print(f"\nTraining for up to {epochs} epochs (early-stop patience=10)...")
        history = train_transformer(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            mlflow_run_id=run_id,
            seed=seed,
        )

        # Final summary metrics.
        mlflow.log_metrics({
            "best_val_loss": min(history["val_loss"]),
            "train_loss_final": history["train_loss"][-1],
            "val_loss_final": history["val_loss"][-1],
            "epochs_trained": len(history["train_loss"]),
        })

        # Save model to the MLflow Model Registry.
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        # Save the fitted scaler as scaler.pkl so evaluate.py can find it by name.
        scaler_tmp = Path(tempfile.mkdtemp()) / "scaler.pkl"
        with open(scaler_tmp, "wb") as f:
            pickle.dump(train_ds.scaler, f)
        mlflow.log_artifact(str(scaler_tmp), artifact_path="")
        scaler_tmp.unlink()

        # ------------------------------------------------------------------
        # Print summary.
        # ------------------------------------------------------------------
        sep = "=" * 60
        print(f"\n{sep}")
        print("Training complete")
        print(f"  Ticker:         {ticker} ({start} -> {end})")
        print(f"  Features:       {len(feature_cols)}")
        print(f"  Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)} samples")
        print(f"  Epochs trained: {len(history['train_loss'])}")
        print(f"  Best val loss:  {min(history['val_loss']):.6f}")
        print(f"  MLflow run ID:  {run_id}")
        print(f"  Model path:     runs:/{run_id}/model")
        print(f"{sep}\n")


if __name__ == "__main__":
    main()
