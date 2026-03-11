"""
data_preprocessing.py — Dataset loading, global preprocessing, and splitting.

Corrected pipeline (four critical fixes applied):

  Fix 1 (Survivorship Bias): Zero-variance columns are identified on the
         RAW train split *including* anomalies, so sensor-features that are
         constant during normal operation but spike during faults are kept.

  Fix 2 (Temporal Continuity): Preprocessing is applied GLOBALLY on the
         continuous timeline.  The sliding window is applied *before*
         anomalous samples are removed from the train set, preventing
         phantom features from stitched non-adjacent time steps.

  Fix 3 (Feature-Space Alignment): The StandardScaler is applied ONLY to
         continuous columns; categorical one-hot columns remain as discrete
         0/1 in both the baseline and VAE branches, ensuring the augmented
         data occupies the same distributional space when combined.

  Corrected order:
    1. Load raw data, sort chronologically, drop leaking columns
    2. Determine chronological split boundaries (6:2:2)
    3. One-hot encode categoricals
    4. Remove zero-variance columns      [fit on RAW train WITH anomalies]
    5. Remove high-NaN columns            [fit on RAW train WITH anomalies]
    6. Scale CONTINUOUS columns only      [fit on RAW train]
    7. Impute (baseline) or NaN→0 (VAE)  [fit on RAW train]
    ── timeline is still continuous here ──
    8. (caller) Apply sliding window GLOBALLY
    9. (caller) Split by boundaries, remove anomalies from train
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ── Data container ──────────────────────────────────────────────────────

@dataclass
class SplitData:
    """Container for a single train/val/test split."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    categorical_indices: List[int] = field(default_factory=list)
    continuous_indices: List[int] = field(default_factory=list)


# ── Loaders ─────────────────────────────────────────────────────────────

def load_secom(data_path: str, label_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the SECOM dataset and sort chronologically by timestamp."""
    X = pd.read_csv(data_path, sep=r"\s+", header=None)

    labels_raw: list[str] = []
    timestamps: list[str] = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: -1 "19/07/2008 11:55:00"
            m = re.match(r"(-?\d+)\s+\"(.+?)\"", line)
            if m:
                labels_raw.append(int(m.group(1)))
                timestamps.append(m.group(2))

    y = pd.Series(labels_raw, name="label")
    ts = pd.to_datetime(timestamps, format="%d/%m/%Y %H:%M:%S")

    # Sort everything chronologically
    sort_idx = ts.argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y.iloc[sort_idx].reset_index(drop=True)

    # Convert labels: 1 → 1 (fail/anomaly), -1 → 0 (pass/normal)
    y = (y == 1).astype(int)
    return X, y


def load_ai4i(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the AI4I 2020 Predictive Maintenance dataset."""
    df = pd.read_csv(data_path)
    y = df["Machine failure"].astype(int)
    # Drop target + leaking columns + ID cols
    drop = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID"]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    return X, y


def load_wafer(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the Wafer Process Quality dataset, sorted by Timestamp."""
    df = pd.read_csv(data_path)

    # Sort chronologically
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    y = df["Defect"].astype(int)
    drop = ["Process_ID", "Wafer_ID", "Defect", "Join_Status", "Timestamp"]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    return X, y


def load_dataset(name: str, cfg: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """Dispatch to the appropriate loader."""
    if name == "SECOM":
        return load_secom(cfg["data_path"], cfg["label_path"])
    elif name == "AI4I_2020":
        return load_ai4i(cfg["data_path"])
    elif name == "Wafer_Quality":
        return load_wafer(cfg["data_path"])
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ── Split boundaries ───────────────────────────────────────────────────

def determine_split_boundaries(
    n: int, train_ratio: float = 0.6, val_ratio: float = 0.2,
) -> Tuple[int, int]:
    """Return ``(train_end, val_end)`` indices for a chronological split."""
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return train_end, val_end


# ── Global preprocessing ──────────────────────────────────────────────

def preprocess_global(
    X: pd.DataFrame,
    train_end: int,
    nan_thresh: float,
    categorical_cols: Optional[List[str]] = None,
    impute_strategy: str = "median",
) -> Tuple[np.ndarray, List[str], List[int], List[int]]:
    """Preprocess the *entire* timeline in one pass.

    All fitters (zero-var, NaN-col, scaler, imputer) are fitted on the
    **raw train split** ``X.iloc[:train_end]`` which *includes* anomalies
    (Fix 1).  Only continuous columns are scaled (Fix 3).

    Parameters
    ----------
    X : DataFrame — full timeline, chronologically sorted.
    train_end : index separating raw-train from val+test.
    nan_thresh : max NaN fraction before a column is dropped.
    categorical_cols : original categorical column names (before one-hot).
    impute_strategy : ``"median"`` for baseline, ``"zero"`` for VAE.

    Returns
    -------
    X_processed : ndarray ``(N, D)``
    feature_names : list of column names after one-hot / pruning
    cat_indices : indices of one-hot categorical columns in the array
    cont_indices : indices of continuous columns in the array
    """
    categorical_cols = categorical_cols or []
    X = X.copy()

    # ── Step 1: One-hot encode categoricals ──
    cat_cols_present = [c for c in categorical_cols if c in X.columns]
    if cat_cols_present:
        X = pd.get_dummies(X, columns=cat_cols_present, dtype=float)

    # ── Reference: raw train split WITH anomalies ──
    X_raw_train = X.iloc[:train_end]

    # ── Step 2: Zero-variance removal (Fix 1 — fit on raw train) ──
    numeric_train = X_raw_train.select_dtypes(include=[np.number])
    zv_cols = list(numeric_train.columns[numeric_train.var(skipna=True) == 0])
    X = X.drop(columns=zv_cols, errors="ignore")

    # ── Step 3: High-NaN removal (fit on raw train) ──
    X_raw_train = X.iloc[:train_end]      # recompute after column drop
    frac = X_raw_train.isna().mean()
    nan_cols = list(frac[frac > nan_thresh].index)
    X = X.drop(columns=nan_cols, errors="ignore")

    feature_names = list(X.columns)

    # ── Identify categorical vs continuous columns ──
    cat_indices: List[int] = []
    cont_indices: List[int] = []
    for i, col in enumerate(feature_names):
        is_cat = any(col.startswith(f"{c}_") for c in categorical_cols)
        if is_cat:
            cat_indices.append(i)
        else:
            cont_indices.append(i)

    arr = X.values.astype(np.float64)

    # ── Step 4: Scale CONTINUOUS columns only (Fix 3) ──
    if cont_indices:
        ci = np.array(cont_indices)
        scaler = StandardScaler()
        scaler.fit(arr[:train_end][:, ci])
        arr[:, ci] = scaler.transform(arr[:, ci])

    # ── Step 5: Impute / fill NaN ──
    if impute_strategy == "median":
        imputer = SimpleImputer(strategy="median")
        imputer.fit(arr[:train_end])
        arr = imputer.transform(arr)
    elif impute_strategy == "zero":
        arr = np.nan_to_num(arr, nan=0.0)

    return arr, feature_names, cat_indices, cont_indices


# ── Post-windowing split ──────────────────────────────────────────────

def split_and_remove_anomalies(
    X_feat: np.ndarray,
    y: np.ndarray,
    train_end: int,
    val_end: int,
    positive_label: int = 1,
) -> SplitData:
    """Split globally-processed (and possibly windowed) data.

    Anomalies in the raw-train portion are removed from train and pushed
    into the validation set, preserving the constraint that the training
    set contains **only** normal samples — without breaking the timeline
    during feature extraction (Fix 2).
    """
    X_tr_raw, y_tr_raw = X_feat[:train_end], y[:train_end]
    X_va_raw, y_va_raw = X_feat[train_end:val_end], y[train_end:val_end]
    X_te, y_te = X_feat[val_end:], y[val_end:]

    normal_mask = y_tr_raw != positive_label
    X_tr = X_tr_raw[normal_mask]
    y_tr = y_tr_raw[normal_mask]

    X_va = np.concatenate([X_tr_raw[~normal_mask], X_va_raw], axis=0)
    y_va = np.concatenate([y_tr_raw[~normal_mask], y_va_raw], axis=0)

    return SplitData(
        X_train=X_tr, X_val=X_va, X_test=X_te,
        y_train=y_tr, y_val=y_va, y_test=y_te,
    )
