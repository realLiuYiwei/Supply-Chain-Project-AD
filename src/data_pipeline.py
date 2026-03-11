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

# Import configuration variables
from config import (
    SECOM_DATA_PATH,
    SECOM_LABEL_PATH,
    AI4I_DATA_PATH,
    WAFER_DATA_PATH,
    RANDOM_STATE
)

# ==========================================
# Preprocessing Modules (解耦出的数据处理模块)
# ==========================================

def preprocess_for_pyod(train_df, val_df, test_df, feature_cols):
    """
    【PyOD 专用流水线】
    职责：过滤无效列 -> 缺失值填充 (Imputation) -> 标准化与编码。
    输出：绝对不含 NaN 的纯净 NumPy 矩阵，可直接喂给 IForest/COPOD 等。
    """
    nunique_counts = train_df[feature_cols].nunique(dropna=True)
    valid_cols = nunique_counts[nunique_counts > 1].index.tolist()
    
    num_cols = train_df[valid_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = train_df[valid_cols].select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ])

    X_train = preprocessor.fit_transform(train_df[valid_cols])
    X_val = preprocessor.transform(val_df[valid_cols])
    X_test = preprocessor.transform(test_df[valid_cols])
    
    return X_train, X_val, X_test, preprocessor

def preprocess_for_vae(train_df, feature_cols):
    """
    【TimeVAE 专用流水线】
    职责：过滤无效列 -> 仅做标准化与编码 -> 绝对不填充 NaN。
    输出：带有 NaN 的 NumPy 矩阵，保留原始缺失分布供 VAE 学习。
    """
    nunique_counts = train_df[feature_cols].nunique(dropna=True)
    valid_cols = nunique_counts[nunique_counts > 1].index.tolist()
    
    num_cols = train_df[valid_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = train_df[valid_cols].select_dtypes(include=['object', 'category']).columns.tolist()

    # 注意这里没有 SimpleImputer
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    X_train_vae = preprocessor.fit_transform(train_df[valid_cols])
    
    return X_train_vae, preprocessor, len(num_cols), len(cat_cols)


# ==========================================
# Data Loading & Splitting Modules 
# ==========================================

def load_secom():
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
    
    train_df = train_df.loc[train_df['label'] == 0].reset_index(drop=True)
    feature_cols = [col for col in data.columns if col not in ['label', 'ts']]
    return "SECOM", train_df, val_df, test_df, feature_cols, "label"

def load_ai4i():
    df = pd.read_csv(AI4I_DATA_PATH)
    df.columns = [col.lower().replace('[', '').replace(']', '').strip() for col in df.columns]
    
    cols_to_drop = ['udi', 'product id', 'twf', 'hdf', 'pwf', 'osf', 'rnf']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    n = len(df)
    train_idx, val_idx = int(n * 0.6), int(n * 0.8)
    
    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()
    
    train_df = train_df.loc[train_df['machine failure'] == 0].reset_index(drop=True)
    feature_cols = [col for col in df.columns if col != 'machine failure']
    return "AI4I_2020", train_df, val_df, test_df, feature_cols, "machine failure"

def load_wafer():
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
    
    train_df = train_df.loc[train_df[target_col] == 0].reset_index(drop=True)
    feature_cols = [col for col in df.columns if col != target_col]
    return "Wafer_Quality", train_df, val_df, test_df, feature_cols, target_col


# ==========================================
# Core Evaluation Engine
# ==========================================

def evaluate_models(X_train, X_test, y_test, dataset_name, exp_type):
    """
    修正 2：极简版模型评估引擎
    职责：只负责接手预处理好的纯净数据并训练/打分。
    注意：这里的 X_train 和 X_test 必须是已经没有 NaN 的 NumPy array。
    """
    models = {
        "IForest": IForest(n_estimators=100, random_state=RANDOM_STATE),
        "COPOD": COPOD(), 
        "ECOD": ECOD(), 
        "LOF": LOF(n_neighbors=20)
    }

    results = []

    for model_name, model in models.items():
        try:
            model.fit(X_train)
            y_test_scores = model.decision_function(X_test)
            
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