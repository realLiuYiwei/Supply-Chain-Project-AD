"""
anomaly_models.py — PyOD anomaly-detection model wrappers.

Supported models: IForest, COPOD, ECOD, LOF.
All models are trained on normal data only and scored on val/test.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF


def get_model(name: str):
    """Instantiate a PyOD model by name."""
    models = {
        "IForest": lambda: IForest(random_state=42),
        "COPOD": lambda: COPOD(),
        "ECOD": lambda: ECOD(),
        "LOF": lambda: LOF(),
    }
    if name not in models:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(models.keys())}"
        )
    return models[name]()


def train_and_score(
    model_name: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a PyOD model on X_train, return anomaly scores for val and test.

    Returns
    -------
    val_scores : (N_val,)  — higher = more anomalous
    test_scores : (N_test,) — higher = more anomalous
    """
    model = get_model(model_name)
    model.fit(X_train)
    val_scores = model.decision_function(X_val)
    test_scores = model.decision_function(X_test)
    return val_scores, test_scores
