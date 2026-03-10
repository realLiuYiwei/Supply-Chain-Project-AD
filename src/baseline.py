import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

# PyOD Models
from pyod.models.iforest import IForest
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.lof import LOF

# Import configuration variables (Ensure these match your actual config.py)
from config import (
    SECOM_DATA_PATH,
    SECOM_LABEL_PATH,
    AI4I_DATA_PATH,
    WAFER_DATA_PATH,
    RANDOM_STATE
)

# ==========================================
# Data Loading & Splitting Modules
# ==========================================

def load_secom():
    """Loads SECOM dataset and splits 6:2:2. Train set contains only normal samples."""
    data = pd.read_csv(SECOM_DATA_PATH, sep=r'\s+', header=None)
    labels = pd.read_csv(SECOM_LABEL_PATH, sep=r'\s+', header=None)
    
    data.columns = [f"feature_{col}" for col in data.columns]
    labels.columns = ['label', "ts"]
    data = pd.concat([labels, data], axis=1)
    
    data["ts"] = pd.to_datetime(data["ts"], dayfirst=True)
    data = data.sort_values('ts').reset_index(drop=True)
    data.loc[data['label'] == -1, 'label'] = 0 
    
    n = len(data)
    train_idx, val_idx = int(n * 0.6), int(n * 0.8)
    
    train_df = data.iloc[:train_idx].copy()
    val_df = data.iloc[train_idx:val_idx].copy()
    test_df = data.iloc[val_idx:].copy()
    
    # Filter Train set to only include normal samples (label == 0)
    train_df = train_df.loc[train_df['label'] == 0].reset_index(drop=True)
    
    feature_cols = [col for col in data.columns if col not in ['label', 'ts']]
    return "SECOM", train_df, val_df, test_df, feature_cols, "label"


def load_ai4i():
    """Loads AI4I dataset and splits 6:2:2. Train set contains only normal samples."""
    df = pd.read_csv(AI4I_DATA_PATH)
    df.columns = [col.lower().replace('[', '').replace(']', '').strip() for col in df.columns]
    
    cols_to_drop = ['udi', 'product id', 'twf', 'hdf', 'pwf', 'osf', 'rnf']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    n = len(df)
    train_idx, val_idx = int(n * 0.6), int(n * 0.8)
    
    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()
    
    # Filter Train set to only include normal samples
    train_df = train_df.loc[train_df['machine failure'] == 0].reset_index(drop=True)
    
    feature_cols = [col for col in df.columns if col != 'machine failure']
    return "AI4I_2020", train_df, val_df, test_df, feature_cols, "machine failure"


def load_wafer():
    """Loads Wafer dataset and splits 6:2:2. Train set contains only normal samples."""
    df = pd.read_csv(WAFER_DATA_PATH)
    df.columns = [col.lower().strip() for col in df.columns]
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    target_col = 'defect' 
    cols_to_drop = ['process_id', 'wafer_id', 'timestamp', 'join_status']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    n = len(df)
    train_idx, val_idx = int(n * 0.6), int(n * 0.8)
    
    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()
    
    # Filter Train set to only include normal samples
    train_df = train_df.loc[train_df[target_col] == 0].reset_index(drop=True)
    
    feature_cols = [col for col in df.columns if col != target_col]
    return "Wafer_Quality", train_df, val_df, test_df, feature_cols, target_col

# ==========================================
# Core Evaluation Engine
# ==========================================

def evaluate_models(train_df, val_df, test_df, feature_cols, target_col, dataset_name, exp_type):
    """
    Evaluates PyOD anomaly detection models using a standard preprocessing pipeline.
    """
    # 1. Filter out features with zero variance in train_df
    nunique_counts = train_df[feature_cols].nunique(dropna=True)
    valid_feature_cols = nunique_counts[nunique_counts > 1].index.tolist()
    
    # 2. Identify numerical and categorical features from the valid columns
    num_features = train_df[valid_feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = train_df[valid_feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

    # 3. Build the preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ]), num_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features)
    ])

    # 4. Fit on Train, Transform Train, Val, and Test
    X_train_processed = preprocessor.fit_transform(train_df[valid_feature_cols])
    X_val_processed = preprocessor.transform(val_df[valid_feature_cols])
    X_test_processed = preprocessor.transform(test_df[valid_feature_cols])
    
    y_test = test_df[target_col].values

    # 5. Initialize Models
    models = {
        "IForest": IForest(n_estimators=100, random_state=RANDOM_STATE),
        "COPOD": COPOD(), 
        "ECOD": ECOD(), 
        "LOF": LOF(n_neighbors=20)
    }

    results = []

    # 6. Train and Evaluate
    for model_name, model in models.items():
        try:
            model.fit(X_train_processed)
            y_test_scores = model.decision_function(X_test_processed)
            
            test_roc = roc_auc_score(y_test, y_test_scores)
            test_pr = average_precision_score(y_test, y_test_scores)
            
            results.append({
                'Dataset': dataset_name,
                'Experiment_Type': exp_type,
                'Model': model_name,
                'ROC-AUC': round(test_roc, 4),
                'PR-AUC': round(test_pr, 4)
            })
        except Exception as e:
            print(f"  [Error] {model_name} failed on {dataset_name}: {e}")
            
    return pd.DataFrame(results)