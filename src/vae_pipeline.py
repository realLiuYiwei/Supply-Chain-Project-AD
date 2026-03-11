"""
vae_pipeline.py — TimeOmniVAE training, generation, and post-processing.

Augmented pipeline (steps 11–14):
  11. Train TimeOmniVAE on training set (normal data only)
  12. Generate synthetic samples  (B, T, D)
  13. Post-process: argmax on categorical one-hot columns,
      then statistical feature aggregation → (B, D_new)
  14. Append to baseline training data
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.time_omni_vae import TimeOmniVAE, TimeOmniVAEConfig, TimeOmniVAETrainer
from src.feature_engineering import sliding_window_features


def build_windowed_tensor(
    X: np.ndarray,
    window_size: int,
) -> torch.Tensor:
    """
    Convert a 2-D array (N, D) into sliding-window sequences (N', T, D)
    for VAE input.

    N' = N - window_size + 1  (only full windows).
    """
    N, D = X.shape
    if window_size <= 1:
        # Each sample is its own 1-step sequence
        return torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, D)

    n_windows = N - window_size + 1
    windows = np.stack(
        [X[i : i + window_size] for i in range(n_windows)], axis=0
    )  # (N', T, D)
    return torch.tensor(windows, dtype=torch.float32)


def _identify_categorical_groups(
    categorical_indices: List[int],
    feature_names: List[str],
    categorical_cols: List[str],
) -> Dict[str, List[int]]:
    """
    Group one-hot column indices by their original categorical column name.

    E.g.  Type_H, Type_L, Type_M  →  {"Type": [idx_H, idx_L, idx_M]}
    """
    groups: Dict[str, List[int]] = {}
    for idx in categorical_indices:
        col_name = feature_names[idx]
        for cat in categorical_cols:
            if col_name.startswith(f"{cat}_"):
                groups.setdefault(cat, []).append(idx)
                break
    return groups


def snap_categorical_argmax(
    generated: np.ndarray,
    categorical_groups: Dict[str, List[int]],
) -> np.ndarray:
    """
    Apply argmax to each group of one-hot columns in the generated data
    to restore discrete categorical states.

    Parameters
    ----------
    generated : (B, T, D)  or  (B, D)
    categorical_groups : mapping  cat_name → list of column indices

    Returns
    -------
    Array with same shape; one-hot columns snapped to 0/1.
    """
    out = generated.copy()
    for _cat_name, indices in categorical_groups.items():
        if generated.ndim == 3:
            vals = out[:, :, indices]  # (B, T, len(indices))
            argmax = vals.argmax(axis=-1)  # (B, T)
            out[:, :, indices] = 0.0
            for k, idx in enumerate(indices):
                out[:, :, idx] = (argmax == k).astype(np.float64)
        else:
            vals = out[:, indices]
            argmax = vals.argmax(axis=-1)
            out[:, indices] = 0.0
            for k, idx in enumerate(indices):
                out[:, idx] = (argmax == k).astype(np.float64)
    return out


def aggregate_generated_features(
    generated: np.ndarray,
) -> np.ndarray:
    """
    Convert generated (B, T, D) → (B, 4D) via statistical aggregation
    (mean, std, min, max)  matching the baseline feature engineering.
    """
    if generated.ndim == 2:
        # Already flat — treat as window_size=1
        B, D = generated.shape
        zeros = np.zeros_like(generated)
        return np.concatenate([generated, zeros, generated, generated], axis=1)

    # (B, T, D)
    feat_mean = generated.mean(axis=1)
    feat_std = generated.std(axis=1, ddof=0)
    feat_min = generated.min(axis=1)
    feat_max = generated.max(axis=1)
    return np.concatenate([feat_mean, feat_std, feat_min, feat_max], axis=1)


def train_vae_and_generate(
    X_train_vae: np.ndarray,
    window_size: int,
    num_samples: int,
    categorical_indices: List[int],
    continuous_indices: List[int],
    feature_names: List[str],
    categorical_cols: List[str],
    *,
    num_clusters: int = 1,
    latent_dim: int = 16,
    rnn_hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    beta: float = 1.0,
    alpha: float = 0.5,
    lambda_temporal: float = 0.1,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda",
) -> np.ndarray:
    """
    Full VAE augmentation: train → generate → post-process → aggregate.

    Parameters
    ----------
    X_train_vae : (N, D) preprocessed training array (NaN-free, one-hot encoded)
    window_size : sliding window T
    num_samples : how many synthetic samples to generate
    categorical_indices : column indices that are one-hot encoded
    continuous_indices  : column indices that are continuous
    feature_names : column names for identifying categorical groups
    categorical_cols : original categorical column names (before one-hot)

    Returns
    -------
    X_augmented : (num_samples, 4*D_baseline)  or  (num_samples, 4*D)
        Aggregated synthetic features ready to append to baseline training data.
    """
    D = X_train_vae.shape[1]

    # Build windowed sequences
    windows = build_windowed_tensor(X_train_vae, window_size)  # (N', T, D)
    dataset = TensorDataset(windows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Configure and build model
    cfg = TimeOmniVAEConfig(
        input_dim=D,
        latent_dim=latent_dim,
        rnn_hidden_dim=rnn_hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        beta=beta,
        alpha=alpha if num_clusters > 1 else 0.0,
        lambda_temporal=lambda_temporal,
    )
    model = TimeOmniVAE(cfg)
    trainer = TimeOmniVAETrainer(
        model=model,
        config=cfg,
        device=device,
        num_clusters=num_clusters,
        lr=lr,
    )

    # Train
    print(f"  [VAE] Training for {epochs} epochs on {len(windows)} windows "
          f"(D={D}, T={window_size}, clusters={num_clusters})...")
    trainer.fit(loader, epochs=epochs, verbose=True)

    # Generate
    print(f"  [VAE] Generating {num_samples} synthetic samples...")
    generated = trainer.generate(num_samples, seq_len=window_size)  # (B, T, D)
    generated = generated.numpy()

    # Post-process: snap categoricals back to discrete
    cat_groups = _identify_categorical_groups(
        categorical_indices, feature_names, categorical_cols
    )
    if cat_groups:
        generated = snap_categorical_argmax(generated, cat_groups)

    # Aggregate to (B, 4D)
    X_augmented = aggregate_generated_features(generated)
    return X_augmented
