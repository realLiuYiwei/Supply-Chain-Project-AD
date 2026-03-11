"""
main.py — Orchestrator for the Supply-Chain Anomaly Detection pipeline.

Corrected pipeline order for each dataset × model combination:

  Baseline:
    1. Load → sort → drop leaking cols
    2. Determine split boundaries (6:2:2)
    3. preprocess_global (Fix 1 + Fix 3: zero-var on raw train, scale continuous only)
    4. Sliding-window features GLOBALLY (Fix 2: continuous timeline)
    5. Split by boundaries, remove anomalies from train → push to val
    6. Train anomaly model → evaluate

  Augmented:
    Same baseline branch for val/test features, plus:
    7. preprocess_global for VAE (NaN→0)
    8. Build VAE windows from GLOBAL continuous timeline
    9. Select normal-train windows (all T samples normal AND within train)
    10. Compute category proportions for conditioned generation (Fix 4)
    11. Train TimeOmniVAE → conditioned generate → snap → aggregate
    12. Combine with baseline train features → train model → evaluate
"""

from __future__ import annotations

import sys
import os
import warnings

import numpy as np

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from src.data_preprocessing import (
    load_dataset,
    determine_split_boundaries,
    preprocess_global,
    split_and_remove_anomalies,
)
from src.feature_engineering import sliding_window_features
from src.vae_pipeline import (
    build_windowed_tensor,
    select_normal_train_windows,
    train_vae_and_generate,
)
from src.anomaly_models import train_and_score
from src.evaluation import collect_results, compute_metrics, print_summary


warnings.filterwarnings("ignore")


def run_baseline_pipeline(
    dataset_name: str,
    dataset_cfg: dict,
    results: list[dict],
) -> None:
    """Execute the baseline pipeline for one dataset."""
    print(f"\n{'─' * 60}")
    print(f"[BASELINE] {dataset_name}")
    print(f"{'─' * 60}")

    # 1. Load + sort + drop
    X, y = load_dataset(dataset_name, dataset_cfg)
    N = len(X)
    print(f"  Loaded: {N} samples, {X.shape[1]} features, "
          f"anomaly rate = {y.mean():.3f}")

    # 2. Split boundaries
    train_end, val_end = determine_split_boundaries(
        N, cfg.TRAIN_RATIO, cfg.VAL_RATIO
    )
    print(f"  Boundaries: train [0, {train_end}), "
          f"val [{train_end}, {val_end}), test [{val_end}, {N})")

    # 3. Global preprocess (Fix 1 + Fix 3)
    X_processed, feat_names, cat_idx, cont_idx = preprocess_global(
        X, train_end,
        nan_thresh=cfg.NAN_COL_REMOVE_THRESH_BASELINE,
        categorical_cols=dataset_cfg["categorical_cols"],
        impute_strategy="median",
    )
    print(f"  After preprocessing: {X_processed.shape[1]} features "
          f"(cont={len(cont_idx)}, cat={len(cat_idx)})")

    # 4. Sliding-window features GLOBALLY (Fix 2)
    X_feat = sliding_window_features(X_processed, cfg.WINDOW_SIZE)
    print(f"  After windowing: {X_feat.shape[1]} features "
          f"(4 × {X_processed.shape[1]})")

    # 5. Split + remove anomalies from train
    y_arr = y.values
    split = split_and_remove_anomalies(
        X_feat, y_arr, train_end, val_end,
        positive_label=dataset_cfg["positive_label"],
    )
    print(f"  Split: train={len(split.X_train)} (all normal), "
          f"val={len(split.X_val)}, test={len(split.X_test)}")

    # 6. Train models
    for model_name in cfg.MODELS:
        print(f"  Training {model_name}...")
        val_scores, test_scores = train_and_score(
            model_name, split.X_train, split.X_val, split.X_test
        )

        val_metrics = compute_metrics(split.y_val, val_scores)
        test_metrics = compute_metrics(split.y_test, test_scores)
        print(f"    Val  ROC-AUC={val_metrics['roc_auc']:.4f}  "
              f"PR-AUC={val_metrics['pr_auc']:.4f}")
        print(f"    Test ROC-AUC={test_metrics['roc_auc']:.4f}  "
              f"PR-AUC={test_metrics['pr_auc']:.4f}")

        for split_name, metrics_dict, y_split in [
            ("val", val_metrics, split.y_val),
            ("test", test_metrics, split.y_test),
        ]:
            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "pipeline": "baseline",
                "split": split_name,
                "roc_auc": metrics_dict["roc_auc"],
                "pr_auc": metrics_dict["pr_auc"],
            })


def run_augmented_pipeline(
    dataset_name: str,
    dataset_cfg: dict,
    results: list[dict],
) -> None:
    """Execute the Time-VAE augmented pipeline for one dataset."""
    print(f"\n{'─' * 60}")
    print(f"[AUGMENTED] {dataset_name}")
    print(f"{'─' * 60}")

    # 1. Load + sort + drop
    X, y = load_dataset(dataset_name, dataset_cfg)
    N = len(X)
    y_arr = y.values
    positive_label = dataset_cfg["positive_label"]

    # 2. Split boundaries
    train_end, val_end = determine_split_boundaries(
        N, cfg.TRAIN_RATIO, cfg.VAL_RATIO
    )

    # ── Baseline branch (for val/test features + baseline train) ───────
    X_base, base_feat_names, base_cat_idx, base_cont_idx = preprocess_global(
        X, train_end,
        nan_thresh=cfg.NAN_COL_REMOVE_THRESH_BASELINE,
        categorical_cols=dataset_cfg["categorical_cols"],
        impute_strategy="median",
    )
    X_base_feat = sliding_window_features(X_base, cfg.WINDOW_SIZE)
    baseline_split = split_and_remove_anomalies(
        X_base_feat, y_arr, train_end, val_end,
        positive_label=positive_label,
    )
    print(f"  Baseline branch: train={len(baseline_split.X_train)} (normal), "
          f"val={len(baseline_split.X_val)}, test={len(baseline_split.X_test)}")

    # ── VAE branch ─────────────────────────────────────────────────────
    X_vae, vae_feat_names, vae_cat_idx, vae_cont_idx = preprocess_global(
        X, train_end,
        nan_thresh=cfg.NAN_COL_REMOVE_THRESH_VAE_INPUT,
        categorical_cols=dataset_cfg["categorical_cols"],
        impute_strategy="zero",
    )
    print(f"  VAE input: {X_vae.shape[1]} features "
          f"(cat={len(vae_cat_idx)}, cont={len(vae_cont_idx)})")

    # Build VAE windows from the GLOBAL continuous timeline (Fix 2)
    T = cfg.WINDOW_SIZE
    vae_windows_tensor = build_windowed_tensor(X_vae, T)
    vae_windows = vae_windows_tensor.numpy()
    n_windows = len(vae_windows)

    # Select normal-train windows (all T samples normal AND inside train)
    normal_train_mask = select_normal_train_windows(
        n_windows, T, y_arr, train_end, positive_label
    )
    train_windows = vae_windows[normal_train_mask]
    print(f"  Normal-train windows: {train_windows.shape[0]} / {n_windows} total")

    # Compute category proportions for conditioned generation (Fix 4)
    num_clusters = dataset_cfg.get("num_clusters", 1)
    cluster_proportions = None

    if num_clusters > 1 and vae_cat_idx:
        # Use row-level (not window-level) proportions from normal-train data
        normal_mask = y_arr[:train_end] != positive_label
        normal_train_rows = X_vae[:train_end][normal_mask]
        cat_proportions = normal_train_rows[:, vae_cat_idx].mean(axis=0)
        total = cat_proportions.sum()
        if total > 0:
            cluster_proportions = cat_proportions / total
        print(f"  Category proportions (Fix 4): {cluster_proportions}")

    n_aug = int(len(train_windows) * cfg.VAE_AUGMENT_MULTIPLIER)

    X_augmented = train_vae_and_generate(
        train_windows=train_windows,
        num_samples=n_aug,
        categorical_indices=vae_cat_idx,
        continuous_indices=vae_cont_idx,
        feature_names=vae_feat_names,
        categorical_cols=dataset_cfg["categorical_cols"],
        cluster_proportions=cluster_proportions,
        num_clusters=num_clusters,
        latent_dim=cfg.VAE_LATENT_DIM,
        rnn_hidden_dim=cfg.VAE_RNN_HIDDEN_DIM,
        num_layers=cfg.VAE_NUM_LAYERS,
        dropout=cfg.VAE_DROPOUT,
        beta=cfg.VAE_BETA,
        alpha=cfg.VAE_ALPHA,
        lambda_temporal=cfg.VAE_LAMBDA_TEMPORAL,
        epochs=cfg.VAE_EPOCHS,
        batch_size=cfg.VAE_BATCH_SIZE,
        lr=cfg.VAE_LR,
        device=cfg.DEVICE,
    )
    print(f"  Generated augmented data: {X_augmented.shape}")

    # Combine: align dimensionality then append
    D_base = baseline_split.X_train.shape[1]
    D_aug = X_augmented.shape[1]

    if D_aug != D_base:
        print(f"  Aligning augmented features ({D_aug}) → baseline ({D_base})")
        if D_aug > D_base:
            X_augmented = X_augmented[:, :D_base]
        else:
            pad = np.zeros((X_augmented.shape[0], D_base - D_aug))
            X_augmented = np.concatenate([X_augmented, pad], axis=1)

    X_train_combined = np.concatenate(
        [baseline_split.X_train, X_augmented], axis=0
    )
    print(f"  Combined training set: {X_train_combined.shape[0]} samples "
          f"({baseline_split.X_train.shape[0]} baseline + "
          f"{X_augmented.shape[0]} synthetic)")

    # Save augmented data
    aug_path = os.path.join(cfg.AUGMENTED_DIR, f"{dataset_name}_augmented.npz")
    np.savez_compressed(aug_path, X_augmented=X_augmented)
    print(f"  Saved augmented data → {aug_path}")

    # Train models
    for model_name in cfg.MODELS:
        print(f"  Training {model_name} (augmented)...")
        val_scores, test_scores = train_and_score(
            model_name, X_train_combined,
            baseline_split.X_val, baseline_split.X_test,
        )

        val_metrics = compute_metrics(baseline_split.y_val, val_scores)
        test_metrics = compute_metrics(baseline_split.y_test, test_scores)
        print(f"    Val  ROC-AUC={val_metrics['roc_auc']:.4f}  "
              f"PR-AUC={val_metrics['pr_auc']:.4f}")
        print(f"    Test ROC-AUC={test_metrics['roc_auc']:.4f}  "
              f"PR-AUC={test_metrics['pr_auc']:.4f}")

        for split_name, metrics_dict in [
            ("val", val_metrics),
            ("test", test_metrics),
        ]:
            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "pipeline": "augmented",
                "split": split_name,
                "roc_auc": metrics_dict["roc_auc"],
                "pr_auc": metrics_dict["pr_auc"],
            })


def main() -> None:
    print("=" * 60)
    print("Supply-Chain Anomaly Detection Pipeline")
    print("=" * 60)

    results: list[dict] = []

    for dataset_name, dataset_cfg in cfg.DATASET_CONFIGS.items():
        # Pipeline 1: Baseline
        run_baseline_pipeline(dataset_name, dataset_cfg, results)

        # Pipeline 2: Augmented with TimeOmniVAE
        run_augmented_pipeline(dataset_name, dataset_cfg, results)

    # Aggregate and save
    save_path = os.path.join(cfg.METRICS_DIR, "pipeline_results.csv")
    df = collect_results(results, save_path)
    print_summary(df)


if __name__ == "__main__":
    main()
