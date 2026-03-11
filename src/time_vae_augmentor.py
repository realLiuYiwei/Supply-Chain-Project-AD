import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset 

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

from time_omni_vae import TimeOmniVAE, TimeOmniVAEConfig, TimeOmniVAETrainer

def create_windows(data: np.ndarray, seq_len: int) -> np.ndarray:
    windows = []
    for i in range(len(data) - seq_len + 1):
        windows.append(data[i : i + seq_len])
    return np.array(windows)

def flatten_windows(windows: np.ndarray) -> np.ndarray:
    return windows[:, -1, :]

def train_and_augment(X_train_vae: np.ndarray, dataset_name: str, preprocessor, num_len: int, latent_dim: int = 16) -> np.ndarray:
    """
    修改点 1：输入直接是预处理好的 NumPy 数组 X_train_vae。
    修改点 2：传入全局 preprocessor 仅仅为了在生成类别特征时进行 One-Hot 的 Argmax 截断（Snapping）。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training TimeOmniVAE on device: {device}")

    num_features = X_train_vae.shape[1]
    cat_len = num_features - num_len
    
    # K 是类别编码后的总维度数，如果没有类别特征则设为 1
    K = cat_len if cat_len > 0 else 1

    # 直接使用预处理好的数据切窗
    windows = create_windows(X_train_vae, SEQ_LEN)

    tensor_windows = torch.tensor(windows, dtype=torch.float32)
    dataset = TensorDataset(tensor_windows)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    config = TimeOmniVAEConfig(
        input_dim=num_features,
        latent_dim=latent_dim,
        lambda_temporal=LAMBDA_TEMPORAL,
        alpha=ALPHA,
        lambda_phys=LAMBDA_PHYS,
        learn_cluster_centers=LEARN_CLUSTER_CENTERS
    )

    model = TimeOmniVAE(config)
    trainer = TimeOmniVAETrainer(
        model=model,
        config=config,
        device=device,
        num_clusters=K
    )

    trainer.fit(loader, epochs=VAE_EPOCHS)

    # 生成数据
    num_gen_windows = len(windows) // 2
    generated_windows = trainer.generate(num_samples=num_gen_windows, seq_len=SEQ_LEN)
    generated_windows = generated_windows.numpy()

    generated_flat = flatten_windows(generated_windows)

    # ==========================================
    # 修改点 3：只做 Snapping (截断)，不做 Inverse Transform
    # 保证输出依然是 One-Hot 格式，方便直接喂给 PyOD
    # ==========================================
    gen_num = generated_flat[:, :num_len]
    
    if cat_len > 0:
        gen_cat_continuous = generated_flat[:, num_len:]
        snapped_cat = np.zeros_like(gen_cat_continuous)
        
        start_idx = 0
        # 从全局预处理器中提取 OneHotEncoder 的类别信息
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot'] if 'cat' in preprocessor.named_transformers_ else preprocessor.named_transformers_['cat']
        
        # 兼容 pipeline 或直接使用 OneHotEncoder 的情况
        categories_list = ohe.categories_ if hasattr(ohe, 'categories_') else []

        for categories in categories_list:
            cat_len_current = len(categories)
            end_idx = start_idx + cat_len_current
            segment = gen_cat_continuous[:, start_idx:end_idx]
            
            # 找到概率最大的类别，将其置为 1，其余为 0
            argmax_indices = np.argmax(segment, axis=1)
            snapped_cat[np.arange(len(segment)), argmax_indices] = 1
            
            start_idx = end_idx

        # 拼接数值和截断后的类别特征
        final_generated = np.hstack([gen_num, snapped_cat])
    else:
        final_generated = gen_num

    # 简单保存为 npy 供后续 debug，或者你可以选择不存
    os.makedirs(AUGMENTED_DIR, exist_ok=True)
    save_path = os.path.join(AUGMENTED_DIR, f"{dataset_name}_augmented_features.npy")
    np.save(save_path, final_generated)
    
    print(f"Successfully generated {len(final_generated)} augmented samples in latent feature space.")
    print(f"Saved NumPy array to: {save_path}")

    # 修改点 4：直接返回 NumPy Array，不用拼 Pandas
    return final_generated

if __name__ == "__main__":
    pass