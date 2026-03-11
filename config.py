"""
config.py — Central configuration for the Supply-Chain Anomaly Detection pipeline.
"""

import os

# =========================================================================
# Directory paths
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DATA_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
AUGMENTED_DIR = os.path.join(BASE_DIR, "augmented_data")
SRC_DIR = os.path.join(BASE_DIR, "src")

# Create directories if they don't exist
for _d in [RESULTS_DIR, METRICS_DIR, AUGMENTED_DIR]:
    os.makedirs(_d, exist_ok=True)

# =========================================================================
# Dataset file paths
# =========================================================================
SECOM_DATA_PATH = os.path.join(BASE_DATA_DIR, "secom", "secom.data")
SECOM_LABEL_PATH = os.path.join(BASE_DATA_DIR, "secom", "secom_labels.data")
AI4I_DATA_PATH = os.path.join(BASE_DATA_DIR, "ai4i_2020", "ai4i2020.csv")
WAFER_DATA_PATH = os.path.join(
    BASE_DATA_DIR, "wafer_process_quality", "semiconductor_quality_control.csv"
)

# =========================================================================
# Preprocessing thresholds
# =========================================================================
NAN_COL_REMOVE_THRESH_BASELINE = 0.50
NAN_COL_REMOVE_THRESH_VAE_INPUT = 0.50

# =========================================================================
# Sliding-window / feature engineering
# =========================================================================
WINDOW_SIZE = 5  # T — kept small since temporal relations are very weak

# =========================================================================
# Data split ratios  (train : val : test = 6 : 2 : 2)
# =========================================================================
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# =========================================================================
# Device
# =========================================================================
DEVICE = "cuda"

# =========================================================================
# Anomaly-detection models (PyOD)
# =========================================================================
MODELS = ["IForest", "COPOD", "ECOD", "LOF"]

# =========================================================================
# Evaluation metrics
# =========================================================================
METRICS = ["roc_auc", "pr_auc"]

# =========================================================================
# TimeOmniVAE hyper-parameters
# =========================================================================
VAE_LATENT_DIM = 16
VAE_RNN_HIDDEN_DIM = 128
VAE_NUM_LAYERS = 2
VAE_DROPOUT = 0.1
VAE_BETA = 1.0
VAE_ALPHA = 0.5
VAE_LAMBDA_TEMPORAL = 0.1
VAE_EPOCHS = 50
VAE_BATCH_SIZE = 64
VAE_LR = 1e-3

# Per-dataset VAE cluster counts (multi-modal joint distribution)
VAE_NUM_CLUSTERS = {
    "SECOM": 1,
    "AI4I_2020": 1,
    "Wafer_Quality": 3,  # lithography / etching / deposition
}

# Number of synthetic samples to generate (multiplier of training-set size)
VAE_AUGMENT_MULTIPLIER = 1.0

# =========================================================================
# Dataset registry
# =========================================================================
DATASET_CONFIGS = {
    "SECOM": {
        "data_path": SECOM_DATA_PATH,
        "label_path": SECOM_LABEL_PATH,
        "target_col": None,  # labels in separate file
        "drop_cols": [],
        "id_cols": [],
        "timestamp_col": None,  # timestamps in label file
        "categorical_cols": [],
        "positive_label": 1,  # 1 = fail, -1 = pass
        "num_clusters": 1,
    },
    "AI4I_2020": {
        "data_path": AI4I_DATA_PATH,
        "label_path": None,
        "target_col": "Machine failure",
        "drop_cols": ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
        "id_cols": ["UDI", "Product ID"],
        "timestamp_col": None,
        "categorical_cols": ["Type"],
        "positive_label": 1,
        "num_clusters": 3,
    },
    "Wafer_Quality": {
        "data_path": WAFER_DATA_PATH,
        "label_path": None,
        "target_col": "Defect",
        "drop_cols": ["Process_ID", "Wafer_ID", "Join_Status"],
        "id_cols": ["Process_ID", "Wafer_ID"],
        "timestamp_col": "Timestamp",
        "categorical_cols": ["Tool_Type"],
        "positive_label": 1,
        "num_clusters": 3,
    },
}
