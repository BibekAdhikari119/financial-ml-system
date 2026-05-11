"""
tests/models/test_transformer.py — Unit tests for PriceTransformer and train_transformer.
"""
from __future__ import annotations

from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.transformer import PriceTransformer, train_transformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader(
    num_batches: int = 3,
    batch_size: int = 4,
    seq_len: int = 10,
    input_dim: int = 5,
) -> DataLoader:
    """Return a DataLoader that yields (X, y) tensors deterministically."""
    torch.manual_seed(0)
    X = torch.randn(num_batches * batch_size, seq_len, input_dim)
    y = torch.randn(num_batches * batch_size, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _make_model(input_dim: int = 5) -> PriceTransformer:
    return PriceTransformer(
        input_dim=input_dim,
        d_model=32,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=64,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_output_shape() -> None:
    """forward() on input (2, 10, 5) must return shape (2, 1)."""
    model = _make_model(input_dim=5)
    x = torch.randn(2, 10, 5)
    out = model(x)
    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"


def test_pos_encoding_not_in_parameters() -> None:
    """pos_enc must be a registered buffer, not a learnable parameter."""
    model = _make_model()
    state_keys = set(model.state_dict().keys())
    param_keys = set(dict(model.named_parameters()).keys())

    assert "pos_enc" in state_keys, "'pos_enc' missing from state_dict()"
    assert "pos_enc" not in param_keys, "'pos_enc' must NOT appear in named_parameters()"


@patch("mlflow.log_metric")
def test_identical_loss_for_same_seed(mock_log_metric) -> None:  # type: ignore[no-untyped-def]
    """Two training runs with identical seeds produce identical first-epoch train losses.

    Both models are constructed from the same random seed so their initial weights
    are identical.  train_transformer then also seeds via torch.manual_seed(seed)
    before the optimiser step, ensuring fully deterministic behaviour.
    """
    loader = _make_loader()

    # Seed model creation identically so initial weights match
    torch.manual_seed(99)
    model_a = _make_model()
    history_a = train_transformer(
        model_a, loader, loader, epochs=1, seed=7
    )

    torch.manual_seed(99)
    model_b = _make_model()
    history_b = train_transformer(
        model_b, loader, loader, epochs=1, seed=7
    )

    assert len(history_a["train_loss"]) == 1
    assert len(history_b["train_loss"]) == 1
    assert history_a["train_loss"][0] == history_b["train_loss"][0], (
        f"Expected identical losses but got "
        f"{history_a['train_loss'][0]} vs {history_b['train_loss'][0]}"
    )


@patch("mlflow.log_metric")
def test_train_runs_two_epochs(mock_log_metric) -> None:  # type: ignore[no-untyped-def]
    """train_transformer completes 2 epochs and returns the expected keys."""
    model = _make_model()
    loader = _make_loader(num_batches=3)

    history = train_transformer(model, loader, loader, epochs=2, seed=42)

    assert "train_loss" in history, "'train_loss' key missing from history"
    assert "val_loss" in history, "'val_loss' key missing from history"
    assert len(history["train_loss"]) == 2, (
        f"Expected 2 train_loss entries, got {len(history['train_loss'])}"
    )
    assert len(history["val_loss"]) == 2, (
        f"Expected 2 val_loss entries, got {len(history['val_loss'])}"
    )
