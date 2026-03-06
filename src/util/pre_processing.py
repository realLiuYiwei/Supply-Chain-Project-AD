import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN

class MissingColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.cols_to_keep = None

    def fit(self, X, y=None):
        missing_ratio = X.isnull().mean()
        cols_below_threshold = missing_ratio[missing_ratio < self.threshold].index
        nunique_counts = X[cols_below_threshold].nunique()
        self.cols_to_keep = nunique_counts[nunique_counts > 1].index.tolist()
        return self

    def transform(self, X):
        return X[self.cols_to_keep]

def get_clean_pipeline(drop_na_col_threshold=0.5, imputer_strategy='median', scaler=StandardScaler()):
    return Pipeline([
        ('drop_invalid_cols', MissingColumnDropper(threshold=drop_na_col_threshold)), 
        ('imputer', SimpleImputer(strategy=imputer_strategy)),             
        ('scaler', scaler)                                
    ])
    
def get_adaptive_eps(df, time_col='ts', quantile=0.95, multiplier=1.5):
    diffs = df[time_col].sort_values().diff().dt.total_seconds().dropna()
    base_eps = diffs.quantile(quantile)
    adaptive_eps = base_eps * multiplier
    return adaptive_eps

def auto_cluster_sessions(df, dataset_name="Dataset"):
    print(f"--- Processing {dataset_name} ---")
    eps = get_adaptive_eps(df)
    ts_numeric = (df['ts'] - df['ts'].min()).dt.total_seconds().values.reshape(-1, 1)
    db = DBSCAN(eps=eps, min_samples=2).fit(ts_numeric)
    df['session_id'] = db.labels_
    n_sessions = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"Auto-detected {n_sessions} sessions with eps={eps/3600:.2f}h")
    return df, eps

def load_secom_data(data_path):
    features = pd.read_csv(data_path / "secom.data", sep=r'\s+', header=None)
    labels = pd.read_csv(data_path / "secom_labels.data", sep=r'\s+', header=None, names=['label', 'ts'])
    
    df = pd.concat([labels, features], axis=1)
    df['ts'] = pd.to_datetime(df['ts'], dayfirst=True)
    df = df.sort_values('ts').reset_index(drop=True)
    df['label'] = df['label'].replace(-1, 0)
    return df


def load_wafer_data(data_path):
    df = pd.read_csv(data_path / "semiconductor_quality_control.csv")
    
    df['ts'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
    df['label'] = df['Defect']
    df.drop(columns=['Timestamp','Defect', 'Join_Status'], inplace=True)
    df.columns = df.columns.str.lower()
    df = df.sort_values('ts').reset_index(drop=True)
    return df