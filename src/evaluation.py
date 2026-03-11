"""
evaluation.py — Metric computation and results persistence.

Metrics:  ROC-AUC,  PR-AUC  (average precision).
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    """
    Compute ROC-AUC and PR-AUC for binary anomaly detection.

    Parameters
    ----------
    y_true  : ground-truth labels (0 = normal, 1 = anomaly)
    scores  : anomaly scores (higher → more anomalous)

    Returns
    -------
    dict with keys 'roc_auc' and 'pr_auc'
    """
    # Guard: if only one class is present, metrics are undefined
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}

    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
    }


def collect_results(
    records: List[Dict],
    save_path: str,
) -> pd.DataFrame:
    """
    Convert a list of result dicts to a DataFrame and save as CSV.

    Each dict should contain at least:
      dataset, model, pipeline, roc_auc, pr_auc
    """
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"  Results saved → {save_path}")
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Pretty-print the results table."""
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A",
        )
    )
    print("=" * 72 + "\n")
