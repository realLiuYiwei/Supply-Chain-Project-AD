import os

# ==========================================
# 1. Directory Paths
# ==========================================
BASE_DATA_DIR = "datasets"
RESULTS_DIR = "results"
METRICS_DIR = "results/metrics"
AUGMENTED_DIR = "augmented_data"
SRC_DIR = "src"

# ==========================================
# 2. Dataset File Paths
# ==========================================
SECOM_DATA_PATH = "datasets/secom/secom.data"
SECOM_LABEL_PATH = "datasets/secom/secom_labels.data"
AI4I_DATA_PATH = "datasets/ai4i_2020/ai4i2020.csv"
WAFER_DATA_PATH = "datasets/wafer_process_quality/semiconductor_quality_control.csv"

# ==========================================
# 3. TimeOmniVAE Specific Design Pattern Hyperparameters
# ==========================================
SEQ_LEN = 12
VAE_EPOCHS = 40
BATCH_SIZE = 64
LAMBDA_TEMPORAL = 0.1      # Weak temporal consistency penalty
ALPHA = 0.5                # Latent clustering penalty for structured modal grouping
LAMBDA_PHYS = 0.0          # Physics limitations penalty, disabled for now
LEARN_CLUSTER_CENTERS = True

# ==========================================
# 4. PyOD Hyperparameters
# ==========================================
IFOREST_N_ESTIMATORS = 100
RANDOM_STATE = 42

# ==========================================
# 5. Automated Directory Initialization
# ==========================================
# Ensure output directories exist before runtime
for directory in [RESULTS_DIR, METRICS_DIR, AUGMENTED_DIR]:
    os.makedirs(directory, exist_ok=True)