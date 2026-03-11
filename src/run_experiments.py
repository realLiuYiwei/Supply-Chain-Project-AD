import pandas as pd
import numpy as np
import os
import logging
from sklearn.impute import SimpleImputer

from config import METRICS_DIR

# 引入解耦后的数据加载、预处理和评估模块
from data_pipeline import (
    load_secom, 
    load_ai4i, 
    load_wafer,
    preprocess_for_pyod,
    preprocess_for_vae,
    evaluate_models
)

# 引入纯张量输出的增强生成模块
from time_vae_augmentor import train_and_augment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_experiments():
    loaders = [
        load_secom,
        load_ai4i,
        load_wafer
    ]
    
    all_results = []
    
    for loader in loaders:
        try:
            # 0. 加载原始数据
            dataset_name, train_df, val_df, test_df, feature_cols, target_col = loader()
            logging.info(f"--- Starting experimental pipeline for {dataset_name} ---")
            
            y_test = test_df[target_col].values

            # ==================================================
            # Step A: Baseline Pipeline (传统 PyOD 纯净流)
            # ==================================================
            logging.info(f"[{dataset_name}] Step A: Running Baseline evaluation.")
            
            # 使用专属预处理器，获得绝对无 NaN 的训练与测试矩阵
            X_train_pyod, X_val_pyod, X_test_pyod, _ = preprocess_for_pyod(
                train_df, val_df, test_df, feature_cols
            )
            
            baseline_metrics = evaluate_models(
                X_train_pyod, X_test_pyod, y_test, 
                dataset_name=dataset_name, 
                exp_type='Baseline'
            )
            all_results.append(baseline_metrics)
            
            # ==================================================
            # Step B: TimeVAE Augmentation Pipeline (带有 NaN 的生成流)
            # ==================================================
            logging.info(f"[{dataset_name}] Step B: Augmenting training data.")
            
            # 使用 VAE 专属预处理器，获得保留了缺失模式的标准化矩阵
            X_train_vae, preprocessor_vae, num_len, cat_len = preprocess_for_vae(
                train_df, feature_cols
            )
            
            # 丢给 VAE 训练并生成，吐出来的直接是带有潜在 NaN 的 NumPy Array
            X_generated = train_and_augment(
                X_train_vae=X_train_vae, 
                dataset_name=dataset_name, 
                preprocessor=preprocessor_vae, 
                num_len=num_len
            )
            
            # ==================================================
            # Step C: Augmented Evaluation (融合与再评估)
            # ==================================================
            logging.info(f"[{dataset_name}] Step C: Running TimeVAE_Augmented evaluation.")
            
            # 1. 垂直拼接原始张量与生成张量
            X_augmented = np.vstack([X_train_vae, X_generated])
            
            # 2. 【防崩溃】虽然 VAE 学习了 NaN，但送给 PyOD 前必须填补！
            # 因为整个矩阵已经是完全的数值型（OneHot编码完成），直接用中位数全局填补即可。
            imputer = SimpleImputer(strategy='median')
            X_augmented_imputed = imputer.fit_transform(X_augmented)
            
            # 3. 进行评估 (注意：为了公平对比，测试集必须是 Step A 里同款的 X_test_pyod)
            augmented_metrics = evaluate_models(
                X_augmented_imputed, X_test_pyod, y_test, 
                dataset_name=dataset_name, 
                exp_type='TimeVAE_Augmented'
            )
            all_results.append(augmented_metrics)
            
            logging.info(f"[{dataset_name}] Pipeline completed successfully.")
            
        except Exception as e:
            logging.error(f"[{dataset_name}] Pipeline failed. Error: {e}")
            continue
            
    if all_results:
        final_results_df = pd.concat(all_results, ignore_index=True)
        os.makedirs(METRICS_DIR, exist_ok=True)
        output_path = os.path.join(METRICS_DIR, 'modular_experiment_results.csv')
        final_results_df.to_csv(output_path, index=False)
        logging.info(f"All experiments finished. Results securely saved to {output_path}")
    else:
        logging.warning("Pipeline finished, but no results were generated. Check dataset loaders.")

if __name__ == "__main__":
    run_experiments()