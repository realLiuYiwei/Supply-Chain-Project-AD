import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset # 修正 1: 导入 PyTorch 数据加载器

from config import (
    SEQ_LEN,
    LAMBDA_TEMPORAL,
    ALPHA,
    LAMBDA_PHYS,
    LEARN_CLUSTER_CENTERS,
    VAE_EPOCHS,
    BATCH_SIZE,
    AUGMENTED_DIR
)

# 修正 2: 显式导入 TimeOmniVAE 本身，以便实例化
from time_omni_vae import TimeOmniVAE, TimeOmniVAEConfig, TimeOmniVAETrainer

def create_windows(data: np.ndarray, seq_len: int) -> np.ndarray:
    windows = []
    for i in range(len(data) - seq_len + 1):
        windows.append(data[i : i + seq_len])
    return np.array(windows)

def flatten_windows(windows: np.ndarray) -> np.ndarray:
    return windows[:, -1, :]

def train_and_augment(train_df: pd.DataFrame, dataset_name: str, num_cols: list, cat_cols: list, latent_dim: int = 16) -> pd.DataFrame:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training TimeOmniVAE on device: {device}")

    scaler = StandardScaler()
    if num_cols:
        scaled_num = scaler.fit_transform(train_df[num_cols])
    else:
        scaled_num = np.empty((len(train_df), 0))

    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cat = ohe.fit_transform(train_df[cat_cols])
        K = encoded_cat.shape[1] 
    else:
        encoded_cat = np.empty((len(train_df), 0))
        K = 1

    processed_data = np.hstack([scaled_num, encoded_cat])
    num_features = processed_data.shape[1]

    windows = create_windows(processed_data, SEQ_LEN)

    # 修正 3: 将 NumPy 数组包装成 PyTorch DataLoader
    tensor_windows = torch.tensor(windows, dtype=torch.float32)
    dataset = TensorDataset(tensor_windows)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 修正 4: 使用正确的 config 参数 (input_dim 而不是 feature_dim)
    config = TimeOmniVAEConfig(
        input_dim=num_features,
        latent_dim=latent_dim,
        lambda_temporal=LAMBDA_TEMPORAL,
        alpha=ALPHA,
        lambda_phys=LAMBDA_PHYS,
        learn_cluster_centers=LEARN_CLUSTER_CENTERS
    )

    # 修正 5: 先实例化模型，再将其传给 Trainer，并移除不支持的 init_centers 等参数
    model = TimeOmniVAE(config)
    trainer = TimeOmniVAETrainer(
        model=model,
        config=config,
        device=device,
        num_clusters=K
    )

    # 修正 6: 传入 DataLoader 和 epoch 数量
    trainer.fit(loader, epochs=VAE_EPOCHS)

    num_gen_windows = len(windows) // 2
    
    # 修正 7: generate 需要传入 seq_len，并转回 NumPy 数组
    generated_windows = trainer.generate(num_samples=num_gen_windows, seq_len=SEQ_LEN)
    generated_windows = generated_windows.numpy()

    generated_flat = flatten_windows(generated_windows)

    num_num_cols = len(num_cols)
    gen_num = generated_flat[:, :num_num_cols]
    gen_cat_continuous = generated_flat[:, num_num_cols:]

    if num_cols:
        restored_num = scaler.inverse_transform(gen_num)
    else:
        restored_num = np.empty((len(generated_flat), 0))

    restored_cat_list = []
    if cat_cols:
        start_idx = 0
        snapped_cat = np.zeros_like(gen_cat_continuous)
        
        for categories in ohe.categories_:
            cat_len = len(categories)
            end_idx = start_idx + cat_len
            segment = gen_cat_continuous[:, start_idx:end_idx]
            argmax_indices = np.argmax(segment, axis=1)
            snapped_cat[np.arange(len(segment)), start_idx + argmax_indices] = 1
            start_idx = end_idx

        restored_cat = ohe.inverse_transform(snapped_cat)
        restored_cat_df = pd.DataFrame(restored_cat, columns=cat_cols)
    else:
        restored_cat_df = pd.DataFrame()

    restored_num_df = pd.DataFrame(restored_num, columns=num_cols)

    generated_df = pd.concat([restored_num_df, restored_cat_df], axis=1)
    generated_df = generated_df[train_df.columns] 

    augmented_train_df = pd.concat([train_df, generated_df], ignore_index=True)

    os.makedirs(AUGMENTED_DIR, exist_ok=True)
    save_path = os.path.join(AUGMENTED_DIR, f"{dataset_name}_augmented_train.csv")
    augmented_train_df.to_csv(save_path, index=False)
    
    print(f"Successfully appended {len(generated_df)} augmented samples.")
    print(f"Saved to: {save_path}")

    return augmented_train_df

if __name__ == "__main__":
    pass