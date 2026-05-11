"""
src/features/dataset.py — PyTorch Dataset for time-series sliding windows.

Provides:
- ``TimeSeriesDataset``: torch Dataset yielding (features, target) windows.
- ``create_train_val_test_splits``: strictly time-ordered split utility.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    pass


class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    Sliding-window Dataset for time-series financial data.

    Each sample is a tuple ``(features, target)`` where:
    - ``features``: ``FloatTensor`` of shape ``(window_size, n_features)``
      containing scaled (or raw) feature values for the window
      ``[t, t + window_size)``.
    - ``target``: ``FloatTensor`` of shape ``(1,)`` containing the forward
      return ``(close[t + window_size - 1 + horizon] - close[t + window_size - 1])
      / close[t + window_size - 1]``.

    Target index alignment
    ----------------------
    For sample at index ``idx``:
    - Feature window covers rows ``idx`` to ``idx + window_size - 1`` (inclusive).
    - The target close at time *t* is ``df["Close"][idx + window_size - 1]``.
    - The target close at time *t + horizon* is
      ``df["Close"][idx + window_size - 1 + horizon]``.

    This means the last valid ``idx`` is ``len(df) - window_size - horizon``,
    so ``__len__`` returns ``len(df) - window_size - horizon``.

    Parameters
    ----------
    df:
        Feature DataFrame.  Must contain all columns in ``feature_cols`` and
        a ``Close`` column for target computation.
    feature_cols:
        List of column names to use as model inputs.
    target_col:
        Unused column name parameter kept for API compatibility.  The target
        is always computed as the forward return of ``Close``.
    window_size:
        Number of time steps per input sample.  Defaults to ``60``.
    horizon:
        Number of steps ahead for the return target.  Defaults to ``1``.
    scaler:
        A fitted ``StandardScaler`` instance (or ``None``).  See
        ``fit_scaler`` for interaction rules.
    fit_scaler:
        If ``True``, a new ``StandardScaler`` is created and fitted on
        ``df[feature_cols]``.  Raises ``ValueError`` if a pre-fitted
        ``scaler`` is also provided (ambiguous intent).

    Raises
    ------
    ValueError
        If ``fit_scaler=True`` and a ``scaler`` is provided simultaneously.
    """

    scaler: StandardScaler | None

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        window_size: int = 60,
        horizon: int = 1,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
    ) -> None:
        if fit_scaler and scaler is not None:
            raise ValueError(
                "Ambiguous scaler configuration: 'fit_scaler=True' was requested "
                "but a pre-fitted 'scaler' was also provided.  Pass either "
                "'fit_scaler=True' (to create and fit a new scaler) or a fitted "
                "'scaler' with 'fit_scaler=False' (to reuse it), not both."
            )

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.window_size = window_size
        self.horizon = horizon

        raw_features: np.ndarray = df[feature_cols].to_numpy(dtype=np.float32)

        # ------------------------------------------------------------------ #
        # Scaler handling                                                       #
        # ------------------------------------------------------------------ #
        if fit_scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(raw_features)
            features_array = self.scaler.transform(raw_features)
        elif scaler is not None:
            self.scaler = scaler
            features_array = self.scaler.transform(raw_features)
        else:
            self.scaler = None
            features_array = raw_features

        self._features: np.ndarray = features_array

        # ------------------------------------------------------------------ #
        # Pre-compute all forward returns at construction time                 #
        # ------------------------------------------------------------------ #
        close_values: np.ndarray = df["Close"].to_numpy(dtype=np.float64)
        n = len(df)
        n_samples = n - window_size - horizon
        targets = np.empty(n_samples, dtype=np.float32)

        for i in range(n_samples):
            t = i + window_size - 1          # last step of the feature window
            t_future = t + horizon           # step at t + horizon
            close_t = close_values[t]
            close_future = close_values[t_future]
            targets[i] = float((close_future - close_t) / close_t) if close_t != 0.0 else 0.0

        self._targets: np.ndarray = targets

    # ---------------------------------------------------------------------- #
    # Dataset protocol                                                         #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Return the number of valid sliding windows."""
        return len(self._targets)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Return the ``idx``-th ``(features, target)`` pair.

        Returns
        -------
        features:
            ``FloatTensor`` of shape ``(window_size, n_features)``.
        target:
            ``FloatTensor`` of shape ``(1,)``.
        """
        feature_window = self._features[idx: idx + self.window_size]  # (W, F)
        target_val = self._targets[idx]                                 # scalar

        features_tensor = torch.as_tensor(feature_window, dtype=torch.float32)
        target_tensor = torch.tensor([target_val], dtype=torch.float32)
        return features_tensor, target_tensor  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Split utility                                                                 #
# --------------------------------------------------------------------------- #

def create_train_val_test_splits(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train, validation, and test partitions.

    Splits are strictly time-ordered (no shuffling).

    Parameters
    ----------
    df:
        Input DataFrame with a time-ordered index.
    train_frac:
        Fraction of rows allocated to the training set.
        Defaults to ``0.7``.
    val_frac:
        Fraction of rows allocated to the validation set.
        Defaults to ``0.15``.  The test set receives the remaining
        ``1 - train_frac - val_frac`` fraction.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)`` — non-overlapping, contiguous,
        time-ordered partitions of *df*.

    Raises
    ------
    ValueError
        If ``train_frac + val_frac >= 1.0`` (no data left for test).
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac ({train_frac}) + val_frac ({val_frac}) must be < 1.0 "
            "so that at least some rows remain for the test set."
        )

    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df
