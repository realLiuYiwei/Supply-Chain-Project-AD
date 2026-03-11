"""
vae_pipeline.py — TimeOmniVAE training, generation, and post-processing.

Fix 4 (Multi-modal Latent Constraint): For multi-cluster datasets (e.g.
Wafer with 3 process stages), generation is *conditioned* on the true
category proportions in the training set.  Instead of uniform cluster
sampling, we:
  1. Encode all normal-train windows to obtain per-sample μ vectors.
  2. Determine each window's category from its one-hot columns.
  3. Compute per-category latent centroid and spread.
  4. Sample categories proportionally, then z = centroid_k + σ_k · ε.
This ensures the generated data respects the real distribution across
process stages even when the learned cluster centres drift.
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


def select_normal_train_windows(
    n_windows: int,
    window_size: int,
    y: np.ndarray,
    train_end: int,
    positive_label: int = 1,
) -> np.ndarray:
    """Return a boolean mask selecting VAE-training windows.

    A window is selected iff:
      • It lies entirely within the raw-train portion (indices 0..train_end-1).
      • **All** samples inside the window are normal.

    Parameters
    ----------
    n_windows : total number of windows built from the global timeline.
    window_size : T.
    y : label array ``(N,)`` — full timeline.
    train_end : first index of the val portion.
    positive_label : label value indicating an anomaly.

    Returns
    -------
    mask : bool array ``(n_windows,)``
    """
    is_normal = (y != positive_label).astype(np.int32)

    if window_size <= 1:
        mask = np.zeros(n_windows, dtype=bool)
        limit = min(train_end, n_windows)
        mask[:limit] = is_normal[:limit].astype(bool)
        return mask

    # Cumsum trick: count normals in each window of size T
    cumsum = np.concatenate([[0], np.cumsum(is_normal)])
    # window i covers original indices [i, i+T-1]
    window_normal_count = cumsum[window_size:] - cumsum[:-window_size]  # (N-T+1,)

    mask = np.zeros(n_windows, dtype=bool)
    # Windows fully inside train: i + T - 1 <= train_end - 1  →  i <= train_end - T
    train_limit = min(train_end - window_size + 1, n_windows)
    if train_limit > 0:
        mask[:train_limit] = (window_normal_count[:train_limit] == window_size)
    return mask


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


# ── Conditioned generation (Fix 4) ────────────────────────────────────

@torch.no_grad()
def conditioned_generate(
    model: TimeOmniVAE,
    train_windows: torch.Tensor,
    cat_group_indices: List[int],
    cluster_proportions: np.ndarray,
    num_samples: int,
    seq_len: int,
    device: str = "cuda",
    noise_std: float = 0.25,
) -> torch.Tensor:
    """Category-conditioned generation for multi-modal datasets.

    Instead of relying on learned cluster centres (which may not align
    with physical process stages), we:
      1. Encode training windows → per-sample μ.
      2. Assign each window's category via majority-vote over its
         one-hot columns across all T time-steps.
      3. Compute per-category latent centroid and spread.
      4. Sample from each category proportionally, z = μ_k + σ_k · ε.

    Parameters
    ----------
    model : trained TimeOmniVAE
    train_windows : (N', T, D) normal-train windows used for training
    cat_group_indices : column indices of the one-hot group (e.g. Tool_Type_*)
    cluster_proportions : (K,) real proportions of each category
    num_samples, seq_len : generation shape parameters
    device, noise_std : generation behaviour
    """
    dev = torch.device(device)
    model.eval()

    # 1. Encode training windows
    x_safe = torch.nan_to_num(train_windows.to(dev), nan=0.0)
    mu, _ = model.encode(x_safe)
    mu = mu.cpu()

    # 2. Determine category per window (majority vote across T steps)
    cat_vals = train_windows[:, :, cat_group_indices].numpy()  # (N', T, K)
    cat_sums = cat_vals.sum(axis=1)                            # (N', K)
    window_cats = cat_sums.argmax(axis=1)                      # (N',)

    K = len(cat_group_indices)

    # 3. Per-category latent stats
    cat_mus: list[torch.Tensor] = []
    cat_stds: list[torch.Tensor] = []
    for c in range(K):
        mask = window_cats == c
        if mask.sum() > 0:
            cat_mus.append(mu[mask].mean(dim=0))
            cat_stds.append(mu[mask].std(dim=0).clamp(min=0.1))
        else:
            cat_mus.append(torch.zeros(mu.shape[1]))
            cat_stds.append(torch.ones(mu.shape[1]) * noise_std)

    cat_mus_t = torch.stack(cat_mus)   # (K, Z)
    cat_stds_t = torch.stack(cat_stds)  # (K, Z)

    # 4. Sample categories proportionally
    props = np.array(cluster_proportions, dtype=np.float64)
    props = props / props.sum()
    assignments = np.random.choice(K, size=num_samples, p=props)

    z_parts: list[torch.Tensor] = []
    for c in range(K):
        count = int((assignments == c).sum())
        if count > 0:
            eps = torch.randn(count, mu.shape[1])
            z_c = cat_mus_t[c] + cat_stds_t[c] * eps
            z_parts.append(z_c)

    z = torch.cat(z_parts, dim=0).to(dev)
    # Shuffle so categories are interleaved
    z = z[torch.randperm(z.size(0))]

    return model.decode(z, seq_len).cpu()


# ── Main entry point ──────────────────────────────────────────────────

def train_vae_and_generate(
    train_windows: np.ndarray,
    num_samples: int,
    categorical_indices: List[int],
    continuous_indices: List[int],
    feature_names: List[str],
    categorical_cols: List[str],
    *,
    cluster_proportions: Optional[np.ndarray] = None,
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
    train_windows : (N_train, T, D) — pre-selected normal-train windows.
    num_samples : how many synthetic samples to generate.
    categorical_indices : column indices that are one-hot encoded.
    continuous_indices  : column indices that are continuous.
    feature_names : column names for identifying categorical groups.
    categorical_cols : original categorical column names (before one-hot).
    cluster_proportions : (K,) real category proportions (Fix 4).

    Returns
    -------
    X_augmented : (num_samples, 4*D)
        Aggregated synthetic features ready to append to baseline training data.
    """
    T = train_windows.shape[1]
    D = train_windows.shape[2]

    windows_tensor = torch.tensor(train_windows, dtype=torch.float32)
    dataset = TensorDataset(windows_tensor)
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
    print(f"  [VAE] Training for {epochs} epochs on {len(train_windows)} windows "
          f"(D={D}, T={T}, clusters={num_clusters})...")
    trainer.fit(loader, epochs=epochs, verbose=True)

    # Generate — conditioned or standard
    cat_groups = _identify_categorical_groups(
        categorical_indices, feature_names, categorical_cols
    )

    use_conditioned = (
        cluster_proportions is not None
        and cat_groups
        and num_clusters > 1
    )

    if use_conditioned:
        first_cat_name = next(iter(cat_groups))
        cat_group_idx = cat_groups[first_cat_name]
        print(f"  [VAE] Conditioned generation using {first_cat_name} proportions "
              f"({dict(zip(range(len(cluster_proportions)), cluster_proportions.round(3)))})")
        generated = conditioned_generate(
            model=model,
            train_windows=windows_tensor,
            cat_group_indices=cat_group_idx,
            cluster_proportions=cluster_proportions,
            num_samples=num_samples,
            seq_len=T,
            device=device,
        )
    else:
        print(f"  [VAE] Standard generation ({num_samples} samples)...")
        generated = trainer.generate(num_samples, seq_len=T)

    generated = generated.numpy()

    # Post-process: snap categoricals back to discrete
    if cat_groups:
        generated = snap_categorical_argmax(generated, cat_groups)

    # Aggregate to (B, 4D)
    X_augmented = aggregate_generated_features(generated)
    return X_augmented
