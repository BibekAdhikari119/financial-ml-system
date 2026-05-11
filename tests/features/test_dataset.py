"""
tests/features/test_dataset.py — Unit tests for TimeSeriesDataset and
create_train_val_test_splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.features.dataset import TimeSeriesDataset, create_train_val_test_splits


# --------------------------------------------------------------------------- #
# Shared fixtures                                                               #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def feature_df() -> pd.DataFrame:
    """
    Synthetic feature DataFrame with 300 rows.

    Columns: Open, High, Low, Close, Volume, sma_10, rsi_14
    Uses seed=42 for reproducibility.
    """
    rng = np.random.default_rng(42)
    n = 300

    returns = rng.normal(0.0, 0.5, size=n)
    close = 100.0 + np.cumsum(returns)
    close = np.clip(close, 1.0, None)

    noise_hi = rng.uniform(0.1, 1.0, size=n)
    noise_lo = rng.uniform(0.1, 1.0, size=n)
    high = close + noise_hi
    low = np.clip(close - noise_lo, 0.01, None)
    open_ = close + rng.uniform(-0.5, 0.5, size=n)
    volume = rng.integers(100_000, 10_000_000, size=n).astype(float)
    sma_10 = pd.Series(close).rolling(10).mean().to_numpy()
    rsi_14 = np.clip(rng.uniform(20, 80, size=n), 0, 100)

    index = pd.date_range("2021-01-04", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "sma_10": sma_10,
            "rsi_14": rsi_14,
        },
        index=index,
    )
    df.index.name = "Date"
    # Drop warm-up NaN rows
    return df.dropna().reset_index(drop=False)


FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "sma_10", "rsi_14"]
WINDOW_SIZE = 30
HORIZON = 1


# --------------------------------------------------------------------------- #
# Test: __getitem__ shape (required)                                            #
# --------------------------------------------------------------------------- #

def test_getitem_shape(feature_df: pd.DataFrame) -> None:
    """
    dataset[0] must return (features, target) with shapes
    (window_size, n_features) and (1,).
    """
    dataset = TimeSeriesDataset(
        df=feature_df,
        feature_cols=FEATURE_COLS,
        target_col="Close",
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
    )
    assert len(dataset) > 0, "Dataset is empty — check fixture size"

    features, target = dataset[0]
    assert isinstance(features, torch.Tensor), "features must be a Tensor"
    assert isinstance(target, torch.Tensor), "target must be a Tensor"
    assert features.shape == (WINDOW_SIZE, len(FEATURE_COLS)), (
        f"Expected features shape ({WINDOW_SIZE}, {len(FEATURE_COLS)}), "
        f"got {tuple(features.shape)}"
    )
    assert target.shape == (1,), (
        f"Expected target shape (1,), got {tuple(target.shape)}"
    )


# --------------------------------------------------------------------------- #
# Test: fit_scaler raises when scaler provided (required)                      #
# --------------------------------------------------------------------------- #

def test_fit_scaler_raises_if_scaler_provided(feature_df: pd.DataFrame) -> None:
    """
    Passing a pre-fitted scaler AND fit_scaler=True must raise ValueError.
    """
    fitted_scaler = StandardScaler()
    fitted_scaler.fit(feature_df[FEATURE_COLS].to_numpy())

    with pytest.raises(ValueError, match="fit_scaler"):
        TimeSeriesDataset(
            df=feature_df,
            feature_cols=FEATURE_COLS,
            target_col="Close",
            window_size=WINDOW_SIZE,
            horizon=HORIZON,
            scaler=fitted_scaler,
            fit_scaler=True,
        )


# --------------------------------------------------------------------------- #
# Test: split non-overlap (required)                                            #
# --------------------------------------------------------------------------- #

def test_split_no_overlap(feature_df: pd.DataFrame) -> None:
    """
    create_train_val_test_splits must produce non-overlapping, time-ordered
    partitions.  Verifies that:
    - last index of train < first index of val
    - last index of val < first index of test (or val is just before test)
    - all three partitions are contiguous (no rows dropped)
    """
    train_df, val_df, test_df = create_train_val_test_splits(feature_df)

    # Sizes should sum to total
    total = len(feature_df)
    assert len(train_df) + len(val_df) + len(test_df) == total, (
        "Split partitions do not sum to the total number of rows"
    )

    # Non-overlapping integer positions (using iloc-based index reset)
    train_last_pos = feature_df.index.get_loc(train_df.index[-1])
    val_first_pos = feature_df.index.get_loc(val_df.index[0])
    val_last_pos = feature_df.index.get_loc(val_df.index[-1])
    test_first_pos = feature_df.index.get_loc(test_df.index[0])

    assert train_last_pos < val_first_pos, (
        "Last train index is not before first val index — overlap detected"
    )
    assert val_last_pos < test_first_pos, (
        "Last val index is not before first test index — overlap detected"
    )

    # Contiguous: val starts immediately after train, test starts after val
    assert val_first_pos == train_last_pos + 1, (
        f"Gap or overlap between train and val: "
        f"train ends at pos {train_last_pos}, val starts at pos {val_first_pos}"
    )
    assert test_first_pos == val_last_pos + 1, (
        f"Gap or overlap between val and test: "
        f"val ends at pos {val_last_pos}, test starts at pos {test_first_pos}"
    )


# --------------------------------------------------------------------------- #
# Test: __len__ formula (required)                                              #
# --------------------------------------------------------------------------- #

def test_len_formula(feature_df: pd.DataFrame) -> None:
    """
    len(dataset) must equal len(df) - window_size - horizon.
    """
    w = 20
    h = 5
    dataset = TimeSeriesDataset(
        df=feature_df,
        feature_cols=FEATURE_COLS,
        target_col="Close",
        window_size=w,
        horizon=h,
    )
    expected_len = len(feature_df) - w - h
    assert len(dataset) == expected_len, (
        f"Expected len(dataset) = {expected_len}, got {len(dataset)}"
    )


# --------------------------------------------------------------------------- #
# Additional tests                                                              #
# --------------------------------------------------------------------------- #

def test_scaler_fit_produces_scaler(feature_df: pd.DataFrame) -> None:
    """When fit_scaler=True, dataset.scaler must be a fitted StandardScaler."""
    dataset = TimeSeriesDataset(
        df=feature_df,
        feature_cols=FEATURE_COLS,
        target_col="Close",
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        fit_scaler=True,
    )
    assert dataset.scaler is not None, "scaler attribute should not be None"
    assert isinstance(dataset.scaler, StandardScaler), (
        f"Expected StandardScaler, got {type(dataset.scaler)}"
    )
    # A fitted scaler has mean_ attribute
    assert hasattr(dataset.scaler, "mean_"), "Scaler does not appear to be fitted"


def test_provided_scaler_not_refitted(feature_df: pd.DataFrame) -> None:
    """When a pre-fitted scaler is passed with fit_scaler=False, it is used as-is."""
    fitted_scaler = StandardScaler()
    fitted_scaler.fit(feature_df[FEATURE_COLS].to_numpy())
    original_mean = fitted_scaler.mean_.copy()

    TimeSeriesDataset(
        df=feature_df,
        feature_cols=FEATURE_COLS,
        target_col="Close",
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        scaler=fitted_scaler,
        fit_scaler=False,
    )
    # The scaler's mean_ must not have changed
    np.testing.assert_array_equal(
        fitted_scaler.mean_,
        original_mean,
        err_msg="Provided scaler was re-fitted inside TimeSeriesDataset",
    )


def test_no_scaling_when_none(feature_df: pd.DataFrame) -> None:
    """When no scaler is provided and fit_scaler=False, features are unscaled."""
    dataset = TimeSeriesDataset(
        df=feature_df,
        feature_cols=FEATURE_COLS,
        target_col="Close",
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        scaler=None,
        fit_scaler=False,
    )
    features, _ = dataset[0]
    # The raw Close values should be in roughly market-price range (>> 1)
    close_idx = FEATURE_COLS.index("Close")
    close_values = features[:, close_idx].numpy()
    assert close_values.mean() > 10.0, (
        "Feature values look scaled when no scaler was requested"
    )


def test_target_is_forward_return(feature_df: pd.DataFrame) -> None:
    """Target value must equal (close[t+horizon] - close[t]) / close[t]."""
    w = 10
    h = 1
    dataset = TimeSeriesDataset(
        df=feature_df,
        feature_cols=FEATURE_COLS,
        target_col="Close",
        window_size=w,
        horizon=h,
    )
    close_values = feature_df["Close"].to_numpy(dtype=np.float64)

    # For idx=0: t = w - 1, t_future = w - 1 + h
    idx = 0
    t = idx + w - 1
    t_future = t + h
    expected = float((close_values[t_future] - close_values[t]) / close_values[t])

    _, target = dataset[idx]
    actual = target.item()
    assert abs(actual - expected) < 1e-5, (
        f"Target mismatch: expected {expected:.6f}, got {actual:.6f}"
    )


def test_split_fractions(feature_df: pd.DataFrame) -> None:
    """Verify approximate partition sizes match the requested fractions."""
    n = len(feature_df)
    train_df, val_df, test_df = create_train_val_test_splits(
        feature_df, train_frac=0.7, val_frac=0.15
    )
    expected_train = int(n * 0.7)
    expected_val = int(n * 0.15)
    expected_test = n - expected_train - expected_val

    assert len(train_df) == expected_train, (
        f"Train size: expected {expected_train}, got {len(train_df)}"
    )
    assert len(val_df) == expected_val, (
        f"Val size: expected {expected_val}, got {len(val_df)}"
    )
    assert len(test_df) == expected_test, (
        f"Test size: expected {expected_test}, got {len(test_df)}"
    )


def test_split_invalid_fracs(feature_df: pd.DataFrame) -> None:
    """create_train_val_test_splits must raise ValueError when fracs sum >= 1.0."""
    with pytest.raises(ValueError):
        create_train_val_test_splits(feature_df, train_frac=0.8, val_frac=0.3)
