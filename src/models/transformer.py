"""
src/models/transformer.py — PyTorch transformer-based price predictor.
"""
from __future__ import annotations

import math

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PriceTransformer(nn.Module):
    """Transformer encoder that predicts the next price from a sequence of features."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        # Build fixed sinusoidal positional encoding: shape (1, max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pos_enc", pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            Tensor of shape (batch, 1) — prediction from the last timestep.
        """
        seq_len = x.size(1)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = x + self.pos_enc[:, :seq_len, :]  # type: ignore[index]
        x = self.encoder(x)  # (batch, seq_len, d_model)
        x = self.output_head(x[:, -1, :])  # (batch, 1)
        return x


def train_transformer(
    model: PriceTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
    mlflow_run_id: str | None = None,
    seed: int = 42,
) -> dict[str, list[float]]:
    """
    Train PriceTransformer with Adam + MSELoss, optional MLflow logging,
    and early stopping (patience = 10 epochs).

    Args:
        model: The PriceTransformer instance to train.
        train_loader: DataLoader yielding (features, targets).
        val_loader: DataLoader yielding (features, targets) — never used for training.
        epochs: Maximum number of training epochs.
        lr: Learning-rate for Adam.
        device: 'cpu' or 'cuda'.
        mlflow_run_id: If provided, logs train_loss and val_loss per epoch.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys 'train_loss' and 'val_loss', each a list of per-epoch means.
    """
    torch.manual_seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_losses: list[float] = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        epoch_train_loss = sum(train_losses) / len(train_losses)

        # --- Validation (no gradient, no weight updates) ---
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                val_preds = model(X_val)
                val_loss = criterion(val_preds, y_val)
                val_losses.append(val_loss.item())

        epoch_val_loss = sum(val_losses) / len(val_losses)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        if mlflow_run_id is not None:
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch, run_id=mlflow_run_id)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch, run_id=mlflow_run_id)

        # --- Early stopping ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return history
