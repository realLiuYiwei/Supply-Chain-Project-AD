import pandas as pd
import os
import logging

from config import METRICS_DIR

# 修正 1: 从 baseline.py 导入正确的加载器和评估函数
from baseline import (
    load_secom, 
    load_ai4i, 
    load_wafer, 
    evaluate_models
)

# 修正 2: 导入正确的生成函数名
from time_vae_augmentor import train_and_augment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_experiments():
    # 修正 3: 使用正确的加载器列表
    loaders = [
        load_secom,
        load_ai4i,
        load_wafer
    ]
    
    all_results = []
    
    for loader in loaders:
        try:
            # 修正 4: 接收 baseline.py 中 load 函数返回的完整元组
            dataset_name, train_df, val_df, test_df, feature_cols, target_col = loader()
            logging.info(f"--- Starting experimental pipeline for {dataset_name} ---")
            
            # Step A: Baseline Evaluation
            logging.info(f"[{dataset_name}] Step A: Running Baseline evaluation.")
            baseline_metrics = evaluate_models(
                train_df, val_df, test_df, feature_cols, target_col, 
                dataset_name=dataset_name, 
                exp_type='Baseline'
            )
            all_results.append(baseline_metrics)
            
            # 修正 5: 在外部提取 num_cols 和 cat_cols 传给 augmentor
            nunique_counts = train_df[feature_cols].nunique(dropna=True)
            valid_feature_cols = nunique_counts[nunique_counts > 1].index.tolist()
            num_cols = train_df[valid_feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = train_df[valid_feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

            # Step B: TimeVAE Augmentation
            logging.info(f"[{dataset_name}] Step B: Augmenting training data.")
            augmented_train_df = train_and_augment(
                train_df, dataset_name, num_cols, cat_cols
            )
            
            # Step C: Augmented Evaluation
            logging.info(f"[{dataset_name}] Step C: Running TimeVAE_Augmented evaluation.")
            augmented_metrics = evaluate_models(
                augmented_train_df, val_df, test_df, feature_cols, target_col, 
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