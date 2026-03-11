"""
feature_engineering.py — Sliding-window statistical feature extraction.

Applies a rolling window of size T over each feature and computes four
summary statistics per feature:  mean, std, min, max.

Output dimensionality:  D_new = 4 * D
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def sliding_window_features(
    X: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Extract statistical features using a sliding window.

    For each sample *i*, the window covers samples [i - window_size + 1, i].
    At the boundaries (i < window_size - 1) the window is left-truncated
    (i.e. we use all available preceding samples).

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input feature matrix (already scaled / imputed).
    window_size : int
        Sliding-window length T.

    Returns
    -------
    X_feat : ndarray of shape (N, 4*D)
        Concatenation of [mean, std, min, max] computed over each window.
    """
    N, D = X.shape
    feat_mean = np.empty((N, D), dtype=np.float64)
    feat_std = np.empty((N, D), dtype=np.float64)
    feat_min = np.empty((N, D), dtype=np.float64)
    feat_max = np.empty((N, D), dtype=np.float64)

    for i in range(N):
        start = max(0, i - window_size + 1)
        window = X[start : i + 1]  # shape (w, D), w <= window_size
        feat_mean[i] = window.mean(axis=0)
        feat_std[i] = window.std(axis=0, ddof=0)
        feat_min[i] = window.min(axis=0)
        feat_max[i] = window.max(axis=0)

    return np.concatenate([feat_mean, feat_std, feat_min, feat_max], axis=1)


def apply_feature_engineering(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply sliding-window feature extraction to all splits independently."""
    return (
        sliding_window_features(X_train, window_size),
        sliding_window_features(X_val, window_size),
        sliding_window_features(X_test, window_size),
    )
