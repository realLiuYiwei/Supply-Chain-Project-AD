import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

# PyOD 模型
from pyod.models.iforest import IForest
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF

# ==========================================
# 1. 配置保存路径 (统一管理)
# ==========================================
BASE_DIR = "results"
PROCESSED_DIR = "processed_data"
PRED_DIR = os.path.join(BASE_DIR, "predictions")
METRIC_DIR = os.path.join(BASE_DIR, "metrics")

for d in [PROCESSED_DIR, PRED_DIR, METRIC_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# 模块 1：数据加载与清洗适配器 (6-2-2 Split)
# ==========================================
def load_secom():
    print("Loading SECOM dataset...")
    # 假设数据在 datasets/secom/ 下
    data = pd.read_csv('datasets/secom/secom.data', sep=r'\s+', header=None)
    labels = pd.read_csv('datasets/secom/secom_labels.data', sep=r'\s+', header=None)
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
    print("Loading AI4I 2020 dataset...")
    df = pd.read_csv("datasets/ai4i_2020/ai4i2020.csv")
    df.columns = [col.lower().replace('[', '').replace(']', '').strip() for col in df.columns]
    cols_to_drop = ['udi', 'product id', 'twf', 'hdf', 'pwf', 'osf', 'rnf']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    n = len(df)
    train_idx, val_idx = int(n * 0.6), int(n * 0.8)
    train_df, val_df, test_df = df.iloc[:train_idx].copy(), df.iloc[train_idx:val_idx].copy(), df.iloc[val_idx:].copy()
    train_df = train_df.loc[train_df['machine failure'] == 0].reset_index(drop=True)
    feature_cols = [col for col in df.columns if col != 'machine failure']
    return "AI4I_2020", train_df, val_df, test_df, feature_cols, "machine failure"

def load_wafer():
    print("Loading Wafer Process Quality dataset...")
    df = pd.read_csv("datasets/wafer_process_quality/semiconductor_quality_control.csv")
    df.columns = [col.lower().strip() for col in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    target_col = 'defect' 
    cols_to_drop = ['process_id', 'wafer_id', 'timestamp', 'join_status']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    n = len(df)
    train_idx, val_idx = int(n * 0.6), int(n * 0.8)
    train_df, val_df, test_df = df.iloc[:train_idx].copy(), df.iloc[train_idx:val_idx].copy(), df.iloc[val_idx:].copy()
    train_df = train_df.loc[train_df[target_col] == 0].reset_index(drop=True)
    feature_cols = [col for col in df.columns if col != target_col]
    return "Wafer_Quality", train_df, val_df, test_df, feature_cols, target_col

# ==========================================
# 模块 2：流水线引擎
# ==========================================
def run_pipeline(dataset_name, train_df, val_df, test_df, feature_cols, target_col):
    print(f"\n---> Executing Pipeline for {dataset_name} <---")
    
    # 1. 过滤无变动特征
    nunique_counts = train_df[feature_cols].nunique(dropna=True)
    unvaried_cols = nunique_counts[nunique_counts <= 1].index.tolist()
    if unvaried_cols:
        feature_cols = [col for col in feature_cols if col not in unvaried_cols]
    
    # 2. 预处理
    num_features = train_df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = train_df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_features)
    ])

    # 3. 转换并保存 .npz
    X_train_processed = preprocessor.fit_transform(train_df[feature_cols])
    X_val_processed = preprocessor.transform(val_df[feature_cols])
    X_test_processed = preprocessor.transform(test_df[feature_cols])

    data_save_path = os.path.join(PROCESSED_DIR, f"{dataset_name}_processed.npz")
    np.savez(data_save_path, 
             X_train=X_train_processed, y_train=train_df[target_col].values,
             X_val=X_val_processed, y_val=val_df[target_col].values,
             X_test=X_test_processed, y_test=test_df[target_col].values)
    
    # 4. 模型评估
    models = {
        "IForest": IForest(n_estimators=100, random_state=42),
        "COPOD": COPOD(), "ECOD": ECOD(), "OCSVM": OCSVM(), "LOF": LOF(n_neighbors=20)
    }

    results = []
    y_test = test_df[target_col].values
    for model_name, model in models.items():
        print(f"  [Training] {model_name}...")
        try:
            model.fit(X_train_processed)
            y_test_scores = model.decision_function(X_test_processed)
            
            test_roc = roc_auc_score(y_test, y_test_scores)
            test_pr = average_precision_score(y_test, y_test_scores)
            
            # 保存预测分
            np.savez(os.path.join(PRED_DIR, f"{dataset_name}_{model_name}_preds.npz"), 
                     y_test=y_test, y_scores=y_test_scores)
            
            results.append({
                'Dataset': dataset_name, 'Model': model_name,
                'Test_ROC-AUC': round(test_roc, 4), 'Test_PR-AUC': round(test_pr, 4)
            })
        except Exception as e:
            print(f"  [Error] {model_name} failed: {e}")
            
    return pd.DataFrame(results)

# ==========================================
# 3. 执行主程序
# ==========================================
if __name__ == "__main__":
    all_results = pd.DataFrame()
    loaders = [load_secom, load_ai4i, load_wafer]
    
    for loader in loaders:
        try:
            ds_name, train_df, val_df, test_df, f_cols, t_col = loader()
            res_df = run_pipeline(ds_name, train_df, val_df, test_df, f_cols, t_col)
            all_results = pd.concat([all_results, res_df], ignore_index=True)
        except Exception as e:
            print(f"Skipping dataset: {e}")
    
    # 【核心修正】：确保 CSV 存在正确的位置
    final_output_path = os.path.join(METRIC_DIR, "baseline_results.csv")
    all_results.to_csv(final_output_path, index=False)
    
    print(f"\n✅ All results saved to: {final_output_path}")
    print(f"✅ Processed data in: {PROCESSED_DIR}")