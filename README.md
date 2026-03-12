# Project Outline: Supply-Chain Anomaly Detection

```text
project_root/
│
├── datasets/                   
├── augmented_data/             
├── results/
│   └── metrics/           
│     
├── src/  
│   ├── data_preprocessing.py   
│   ├── feature_engineering.py  
│   ├── vae_pipeline.py         
│   ├── anomaly_models.py       
│   ├── evaluation.py           
│   └── time_omni_vae.py        
│
├── config.py       
├── requirement.txt
└── main.py                 

```

## Time-VAE Generator Configuration Settings

* **NaN-col-remove_thresh_baseline**: 50%
* **NaN-col-remove_thresh_vae_input**: 50%
* **Window_Size ($T$)**: Recommended to be set small, since temporal relations are very weak.

### 1. Directory Paths

* `BASE_DATA_DIR` = "datasets"
* `RESULTS_DIR` = "results"
* `METRICS_DIR` = "results/metrics"
* `AUGMENTED_DIR` = "augmented_data"
* `SRC_DIR` = "src"

### 2. Dataset File Paths

* `SECOM_DATA_PATH` = "datasets/secom/secom.data"
* `SECOM_LABEL_PATH` = "datasets/secom/secom_labels.data"
* `AI4I_DATA_PATH` = "datasets/ai4i_2020/ai4i2020.csv"
* `WAFER_DATA_PATH` = "datasets/wafer_process_quality/semiconductor_quality_control.csv"

### 3. Hyperparameters & Settings

* **Device used**: cuda
* **Models used**: [IForest, COPOD, ECOD, LOF]
* **Metrics**: pr-auc, roc-auc
* **Data Split**: train:val:test = 6:2:2
* **Constraint**: Train dataset **MUST** consist entirely of normal data (no defects).

---

## 4. Baseline Training Pipeline

1. **Load Raw Data**.
2. **Sort chronologically** by timestamp (crucial for SECOM).
3. **Drop target-leaking columns** (e.g., TWF, HDF in AI4I).
4. **Determine Split Boundaries**: Determine the 6:2:2 Train/Val/Test split boundaries **FIRST**.
5. **Encode Categoricals**: Apply One-Hot encoding to discrete categorical columns.
6. **Remove 0-variance columns**: [*Fit ONLY on RAW Train WITH anomalies*].
7. **Remove High-NaN columns**: Remove columns with NaN over `Nan-col-remove_thresh` [*Fit ONLY on RAW Train WITH anomalies*].
8. **Scale Features**: Scaler [*Fit ONLY on NORMAL Train to avoid masking effects from outliers, transform on Val/Test. Applies to CONTINUOUS columns only*].
9. **Impute Missing Values**: Imputer [*Fit ONLY on Train, transform on Val/Test*].
10. **Extract Features**: Apply sliding window and extract statistical features GLOBALLY on the continuous timeline ($D_{new} = 4D$).
11. **Clean Train Set**: Split by boundaries and remove anomalies from the Train set. *(Pushing them to Validation by concatenating them at the head of the Val set; note this breaks strict temporal continuity for Val, but is valid for point-wise scoring models)*.
12. **Train model**.

---

## 5. Time-VAE Augmented Training Pipeline

1. **Load Raw Data**.
2. **Sort chronologically** by timestamp.
3. **Drop target-leaking columns**.
4. **Determine Split Boundaries**: Determine the 6:2:2 Train/Val/Test split boundaries **FIRST**.
5. **Encode Categoricals**: Apply One-Hot encoding to discrete categorical columns (for the VAE to process).
6. **Remove 0-variance columns**: [*Fit ONLY on RAW Train WITH anomalies*].
7. **Remove High-NaN columns**: Remove columns with NaN over `Nan-col-remove_thresh_vae_input` [*Fit ONLY on RAW Train WITH anomalies*].
8. **Scale Features**: Scaler [*Fit ONLY on NORMAL Train, transform on Val/Test. Applies to CONTINUOUS columns only*].
9. **Handle NaNs for VAE**: NaN replacement (fill with 0.0) [*fit on RAW train*].
10. **Build VAE Windows**: Build from the GLOBAL continuous timeline.
11. **Filter Windows**: Select normal-train windows. *(Ensure all $T$ samples in the window are normal AND within the train boundary)*.
12. **Conditioned Setup**: Compute category proportions from normal-train data for conditioned generation (using row-level means or majority vote across $T$ steps).
13. **Train Time-VAE**: Time-VAE learns from the data. [*Train ONLY on the Train set to learn normal distribution; for Wafer dataset, set `num_clusters=3*`].
14. **Generate Data**: Generate augmented data (Shape: $B$, $T$, $D$) conditioned on category proportions.
15. **Output Post-processing (Addition)**: Apply argmax on the generated data to restore categorical columns to discrete states, then execute statistical feature aggregation to convert dimensions to ($B$, $D_{new}$).
16. **Align Features**: Align dimensionality of augmented features with baseline features. *(Require exact column matching via `feature_names` to prevent feature misalignment, rather than simple right-side padding/truncation)*.
17. **Merge Data**: Combine with processed baseline data. *(Append the generated normal data to the Train set)*.
18. **Train model**.

*(Note: Temporal relations are very weak in my used 3 datasets.)*

---

## 6. Dataset Specifics & Action Items

**Centralized Configuration Enforcement:** All dataset specifics (target columns, drop columns, timestamp logic) **MUST** be read dynamically from the `DATASET_CONFIGS` registry in `config.py` rather than being hardcoded in the loader functions.

### SECOM Dataset

* **Details:** All features are numerical. Labels and timestamps are located in `SECOM_LABEL_PATH` in the following format:
```text
-1 "19/07/2008 11:55:00"
-1 "19/07/2008 12:32:00"
1 "19/07/2008 13:17:00"

```


*(No column names are provided.)*
* **Action:** After loading the labels file, you **MUST** strictly sort the features chronologically based on the timestamps before applying the sliding window and Train/Test split.

### Wafer Process Quality Dataset

* **Details:** This dataset contains 4,219 real-time sensor readings simulated to reflect advanced semiconductor fabrication processes—specifically during lithography, etching, and deposition. Each row corresponds to a unique wafer processing instance, with measurements from critical tools and sensors. These readings determine if a wafer is a Joining wafer (successful) or a Non-Joining wafer (defective/rejected).
* **Non-Joint Modelling:** Performed independently to predict quality control (fault prediction) without prior filtering of faulty wafers.
* **Joint Modelling:** A two-step model that first detects faulty wafers via classification, then performs prediction ONLY on normal wafers to improve accuracy and reduce noise.
* **Process Stages:** `["lithography", "etching", "deposition"]`
*(Note: Sensor reading distributions are NOT expected to be unified across the 3 modes.)*
* **Columns:** Process_ID, Timestamp, Tool_Type, Wafer_ID, Chamber_Temperature, Gas_Flow_Rate, RF_Power, Etch_Depth, Rotation_Speed, Vacuum_Pressure, Stage_Alignment_Error, Vibration_Level, UV_Exposure_Intensity, Particle_Count, Target = Defect, Join_Status (remove).
* **Action:** Because there are 3 distinct process stages with varying distributions, you **MUST** pass `num_clusters = 3` when instantiating the TimeOmniVAE to activate multi-modal joint distribution clustering in the latent space.

### AI4I 2020 Dataset

* **Details:** The only target we need to focus on is Machine failure.

| Variable Name | Role | Type | Description / Units | Missing Values |
| --- | --- | --- | --- | --- |
| **UID** | ID | Integer | - | no |
| **Product ID** | ID | Categorical | - | no |
| **Type** | Feature | Categorical | - | no |
| **Air temperature** | Feature | Continuous | K | no |
| **Process temperature** | Feature | Continuous | K | no |
| **Rotational speed** | Feature | Integer | rpm | no |
| **Torque** | Feature | Continuous | Nm | no |
| **Tool wear** | Feature | Integer | min | no |
| **Machine failure** | Target | Integer | - | no |
| **TWF** | Target | Integer | - | no |
| **HDF** | Target | Integer | - | no |
| **PWF** | Target | Integer | - | no |
| **OSF** | Target | Integer | - | no |
| **RNF** | Target | Integer | - | no |

* **Action:** During preprocessing, immediately drop the TWF, HDF, PWF, OSF, and RNF columns to prevent data leakage (stopping the downstream models and VAE from "peeking" at the answers). One-Hot encode the `Type` column before it enters the VAE, apply argmax immediately after generation to restore its discrete state, and then merge it back with the baseline data.
