"""
main.py — Orchestrator for the Supply-Chain Anomaly Detection pipeline.

Runs two pipelines for each dataset × model combination:
  1. Baseline   — raw → preprocess → feature engineering → model → evaluate
  2. Augmented  — raw → preprocess(VAE) → TimeOmniVAE → post-process →
                  combine with baseline → model → evaluate
"""

from __future__ import annotations

import sys
import os
import copy
import warnings

import numpy as np

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from src.data_preprocessing import (
    load_dataset,
    split_train_val_test,
    preprocess_baseline,
    preprocess_for_vae,
)
from src.feature_engineering import apply_feature_engineering
from src.vae_pipeline import train_vae_and_generate
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

    # 1–3. Load + sort + drop
    X, y = load_dataset(dataset_name, dataset_cfg)
    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features, "
          f"anomaly rate = {y.mean():.3f}")

    # 4. Split (train = ONLY normal)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
        positive_label=dataset_cfg["positive_label"],
    )
    print(f"  Split: train={len(X_train)} (all normal), "
          f"val={len(X_val)}, test={len(X_test)}")

    # 5–8. Preprocess
    split = preprocess_baseline(
        X_train.copy(), X_val.copy(), X_test.copy(),
        y_train, y_val, y_test,
        nan_thresh=cfg.NAN_COL_REMOVE_THRESH_BASELINE,
        categorical_cols=dataset_cfg["categorical_cols"],
    )
    print(f"  After preprocessing: {split.X_train.shape[1]} features")

    # 9. Feature engineering
    X_tr_feat, X_va_feat, X_te_feat = apply_feature_engineering(
        split.X_train, split.X_val, split.X_test,
        window_size=cfg.WINDOW_SIZE,
    )
    print(f"  After feature engineering: {X_tr_feat.shape[1]} features "
          f"(4 × {split.X_train.shape[1]})")

    # 10–11. Train models
    for model_name in cfg.MODELS:
        print(f"  Training {model_name}...")
        val_scores, test_scores = train_and_score(
            model_name, X_tr_feat, X_va_feat, X_te_feat
        )

        val_metrics = compute_metrics(split.y_val, val_scores)
        test_metrics = compute_metrics(split.y_test, test_scores)
        print(f"    Val  ROC-AUC={val_metrics['roc_auc']:.4f}  "
              f"PR-AUC={val_metrics['pr_auc']:.4f}")
        print(f"    Test ROC-AUC={test_metrics['roc_auc']:.4f}  "
              f"PR-AUC={test_metrics['pr_auc']:.4f}")

        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "pipeline": "baseline",
            "split": "val",
            "roc_auc": val_metrics["roc_auc"],
            "pr_auc": val_metrics["pr_auc"],
        })
        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "pipeline": "baseline",
            "split": "test",
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
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

    # 1–3. Load + sort + drop
    X, y = load_dataset(dataset_name, dataset_cfg)

    # 4. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
        positive_label=dataset_cfg["positive_label"],
    )
    print(f"  Split: train={len(X_train)} (all normal), "
          f"val={len(X_val)}, test={len(X_test)}")

    # ── Baseline branch (for val/test features & combining later) ───────
    baseline_split = preprocess_baseline(
        X_train.copy(), X_val.copy(), X_test.copy(),
        y_train, y_val, y_test,
        nan_thresh=cfg.NAN_COL_REMOVE_THRESH_BASELINE,
        categorical_cols=dataset_cfg["categorical_cols"],
    )
    X_tr_feat_base, X_va_feat, X_te_feat = apply_feature_engineering(
        baseline_split.X_train, baseline_split.X_val, baseline_split.X_test,
        window_size=cfg.WINDOW_SIZE,
    )

    # ── VAE branch (steps 5–13) ────────────────────────────────────────
    vae_split, cat_idx, cont_idx = preprocess_for_vae(
        X_train.copy(), X_val.copy(), X_test.copy(),
        y_train, y_val, y_test,
        nan_thresh=cfg.NAN_COL_REMOVE_THRESH_VAE_INPUT,
        categorical_cols=dataset_cfg["categorical_cols"],
    )
    print(f"  VAE input: {vae_split.X_train.shape[1]} features "
          f"(cat={len(cat_idx)}, cont={len(cont_idx)})")

    num_clusters = dataset_cfg.get("num_clusters", 1)
    n_aug = int(len(vae_split.X_train) * cfg.VAE_AUGMENT_MULTIPLIER)

    X_augmented = train_vae_and_generate(
        X_train_vae=vae_split.X_train,
        window_size=cfg.WINDOW_SIZE,
        num_samples=n_aug,
        categorical_indices=cat_idx,
        continuous_indices=cont_idx,
        feature_names=vae_split.feature_names,
        categorical_cols=dataset_cfg["categorical_cols"],
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

    # 14. Combine: match dimensionality with baseline features
    # The baseline X_tr_feat_base has shape (N_train, 4*D_baseline).
    # The augmented X_augmented has shape (n_aug, 4*D_vae).
    # D_vae and D_baseline may differ due to different NaN thresholds or
    # one-hot encoding.  We align by padding/truncating to match baseline.
    D_base = X_tr_feat_base.shape[1]
    D_aug = X_augmented.shape[1]

    if D_aug != D_base:
        print(f"  Aligning augmented features ({D_aug}) → baseline ({D_base})")
        if D_aug > D_base:
            X_augmented = X_augmented[:, :D_base]
        else:
            pad = np.zeros((X_augmented.shape[0], D_base - D_aug))
            X_augmented = np.concatenate([X_augmented, pad], axis=1)

    X_train_combined = np.concatenate([X_tr_feat_base, X_augmented], axis=0)
    print(f"  Combined training set: {X_train_combined.shape[0]} samples "
          f"({X_tr_feat_base.shape[0]} baseline + {X_augmented.shape[0]} synthetic)")

    # Save augmented data
    aug_path = os.path.join(
        cfg.AUGMENTED_DIR, f"{dataset_name}_augmented.npz"
    )
    np.savez_compressed(aug_path, X_augmented=X_augmented)
    print(f"  Saved augmented data → {aug_path}")

    # 15. Train models
    for model_name in cfg.MODELS:
        print(f"  Training {model_name} (augmented)...")
        val_scores, test_scores = train_and_score(
            model_name, X_train_combined, X_va_feat, X_te_feat
        )

        val_metrics = compute_metrics(baseline_split.y_val, val_scores)
        test_metrics = compute_metrics(baseline_split.y_test, test_scores)
        print(f"    Val  ROC-AUC={val_metrics['roc_auc']:.4f}  "
              f"PR-AUC={val_metrics['pr_auc']:.4f}")
        print(f"    Test ROC-AUC={test_metrics['roc_auc']:.4f}  "
              f"PR-AUC={test_metrics['pr_auc']:.4f}")

        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "pipeline": "augmented",
            "split": "val",
            "roc_auc": val_metrics["roc_auc"],
            "pr_auc": val_metrics["pr_auc"],
        })
        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "pipeline": "augmented",
            "split": "test",
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
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
