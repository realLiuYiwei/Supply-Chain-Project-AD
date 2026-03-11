"""
data_preprocessing.py — Dataset loading, cleaning, and train/val/test splitting.

Pipeline order (both baseline and VAE):
  1. Load raw data
  2. Sort chronologically by timestamp (crucial for SECOM)
  3. Drop target-leaking / ID columns
  4. 6:2:2 train/val/test split (train = ONLY normal data)
  5. Remove zero-variance columns      [fit on train]
  6. Remove high-NaN columns            [fit on train]
  7. Scale                              [fit on train]
  8. Impute (baseline) or fill 0 (VAE)  [fit on train]
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


# ── Splitting ───────────────────────────────────────────────────────────

def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    *,
    positive_label: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Split data 6:2:2 with the constraint that training set contains ONLY
    normal samples.  Anomalous samples from what would have been the training
    portion are pushed into the validation set.

    Data is NOT shuffled — temporal ordering is preserved.
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train_raw, y_train_raw = X.iloc[:train_end], y.iloc[:train_end]
    X_val_raw, y_val_raw = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    # Move anomalies OUT of training set → into validation set
    normal_mask = y_train_raw != positive_label
    X_train = X_train_raw[normal_mask].reset_index(drop=True)
    y_train = y_train_raw[normal_mask].reset_index(drop=True)

    anomalous_from_train = X_train_raw[~normal_mask]
    anomalous_labels = y_train_raw[~normal_mask]

    X_val = pd.concat([anomalous_from_train, X_val_raw], ignore_index=True)
    y_val = pd.concat([anomalous_labels, y_val_raw], ignore_index=True)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Column pruning ──────────────────────────────────────────────────────

def get_zero_variance_cols(X_train: pd.DataFrame) -> List[str]:
    """Identify columns with zero variance (fit on train only)."""
    numeric = X_train.select_dtypes(include=[np.number])
    return list(numeric.columns[numeric.var(skipna=True) == 0])


def get_high_nan_cols(
    X_train: pd.DataFrame, threshold: float
) -> List[str]:
    """Columns where NaN fraction > threshold (fit on train only)."""
    frac = X_train.isna().mean()
    return list(frac[frac > threshold].index)


# ── Preprocessing pipeline ──────────────────────────────────────────────

def preprocess_baseline(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    nan_thresh: float,
    categorical_cols: Optional[List[str]] = None,
) -> SplitData:
    """
    Baseline preprocessing (steps 5–8):
      5. Remove zero-variance columns
      6. Remove high-NaN columns
      7. StandardScaler
      8. SimpleImputer (median)

    One-hot encodes categorical columns first, then applies numeric pipeline.
    """
    categorical_cols = categorical_cols or []

    # One-hot encode categoricals before numeric pipeline
    if categorical_cols:
        cats_in_train = [c for c in categorical_cols if c in X_train.columns]
        X_train = pd.get_dummies(X_train, columns=cats_in_train, dtype=float)
        X_val = pd.get_dummies(X_val, columns=cats_in_train, dtype=float)
        X_test = pd.get_dummies(X_test, columns=cats_in_train, dtype=float)
        # Align columns (train is reference)
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0.0)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    # Step 5 — zero-variance
    zv_cols = get_zero_variance_cols(X_train)
    X_train = X_train.drop(columns=zv_cols, errors="ignore")
    X_val = X_val.drop(columns=zv_cols, errors="ignore")
    X_test = X_test.drop(columns=zv_cols, errors="ignore")

    # Step 6 — high-NaN
    nan_cols = get_high_nan_cols(X_train, nan_thresh)
    X_train = X_train.drop(columns=nan_cols, errors="ignore")
    X_val = X_val.drop(columns=nan_cols, errors="ignore")
    X_test = X_test.drop(columns=nan_cols, errors="ignore")

    feature_names = list(X_train.columns)

    # Step 7 — scale
    scaler = StandardScaler()
    arr_train = scaler.fit_transform(X_train.values.astype(np.float64))
    arr_val = scaler.transform(X_val.values.astype(np.float64))
    arr_test = scaler.transform(X_test.values.astype(np.float64))

    # Step 8 — impute
    imputer = SimpleImputer(strategy="median")
    arr_train = imputer.fit_transform(arr_train)
    arr_val = imputer.transform(arr_val)
    arr_test = imputer.transform(arr_test)

    return SplitData(
        X_train=arr_train,
        X_val=arr_val,
        X_test=arr_test,
        y_train=y_train.values,
        y_val=y_val.values,
        y_test=y_test.values,
        feature_names=feature_names,
    )


def preprocess_for_vae(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    nan_thresh: float,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[SplitData, List[int], List[int]]:
    """
    VAE preprocessing (steps 5–10):
      5. Remove zero-variance columns
      6. Remove high-NaN columns (using VAE threshold)
      7. StandardScaler  (on continuous columns)
      8. NaN → 0.0
      9. NaN mask creation  (stored in SplitData arrays as extra info)
     10. One-hot encode categoricals

    Returns
    -------
    split : SplitData — arrays with NaN replaced by 0, categoricals one-hot
    categorical_indices : column indices that are one-hot categorical
    continuous_indices  : column indices that are continuous
    """
    categorical_cols = categorical_cols or []

    # Step 5 — zero-variance
    zv_cols = get_zero_variance_cols(X_train)
    X_train = X_train.drop(columns=zv_cols, errors="ignore")
    X_val = X_val.drop(columns=zv_cols, errors="ignore")
    X_test = X_test.drop(columns=zv_cols, errors="ignore")

    # Step 6 — high-NaN (VAE threshold)
    nan_cols = get_high_nan_cols(X_train, nan_thresh)
    X_train = X_train.drop(columns=nan_cols, errors="ignore")
    X_val = X_val.drop(columns=nan_cols, errors="ignore")
    X_test = X_test.drop(columns=nan_cols, errors="ignore")

    # Separate continuous and categorical
    cat_cols_present = [c for c in categorical_cols if c in X_train.columns]
    cont_cols = [c for c in X_train.columns if c not in cat_cols_present]

    # Step 7 — scale continuous columns only
    scaler = StandardScaler()
    if cont_cols:
        X_train[cont_cols] = scaler.fit_transform(
            X_train[cont_cols].values.astype(np.float64)
        )
        X_val[cont_cols] = scaler.transform(
            X_val[cont_cols].values.astype(np.float64)
        )
        X_test[cont_cols] = scaler.transform(
            X_test[cont_cols].values.astype(np.float64)
        )

    # Step 8 — NaN → 0.0 (after scaling, so 0 is meaningful)
    X_train = X_train.fillna(0.0)
    X_val = X_val.fillna(0.0)
    X_test = X_test.fillna(0.0)

    # Step 10 — one-hot encode categoricals
    if cat_cols_present:
        X_train = pd.get_dummies(X_train, columns=cat_cols_present, dtype=float)
        X_val = pd.get_dummies(X_val, columns=cat_cols_present, dtype=float)
        X_test = pd.get_dummies(X_test, columns=cat_cols_present, dtype=float)
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0.0)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    feature_names = list(X_train.columns)

    # Identify categorical vs continuous indices in the final array
    categorical_indices: List[int] = []
    continuous_indices: List[int] = []
    for i, col in enumerate(feature_names):
        is_cat = any(col.startswith(f"{c}_") for c in categorical_cols)
        if is_cat:
            categorical_indices.append(i)
        else:
            continuous_indices.append(i)

    split = SplitData(
        X_train=X_train.values.astype(np.float64),
        X_val=X_val.values.astype(np.float64),
        X_test=X_test.values.astype(np.float64),
        y_train=y_train.values,
        y_val=y_val.values,
        y_test=y_test.values,
        feature_names=feature_names,
        categorical_indices=categorical_indices,
        continuous_indices=continuous_indices,
    )
    return split, categorical_indices, continuous_indices
