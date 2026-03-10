import pandas as pd
import os
import logging

# 1. Import METRICS_DIR from config
from config import METRICS_DIR

# 2. Import loaders and evaluate_models from baseline_pipeline
from baseline_pipeline import (
    load_sensor_data, 
    load_qc_data, 
    load_machine_telemetry_data, 
    evaluate_models
)

# 2 (cont). Import the generation function from timevae_augmentor
from time_vae_augmentor import augment_with_timevae

# Configure basic logging for observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_experiments():
    # 3. Iterate over the three dataset loaders
    loaders = [
        load_sensor_data,
        load_qc_data,
        load_machine_telemetry_data
    ]
    
    all_results = []
    
    for loader in loaders:
        dataset_name = loader.__name__
        logging.info(f"--- Starting experimental pipeline for {dataset_name} ---")
        
        # 4. Wrap in a try-except block to ensure pipeline continuity
        try:
            # Load Train, Val, Test DataFrames
            train_df, val_df, test_df = loader()
            
            # Step A: Baseline Evaluation
            logging.info(f"[{dataset_name}] Step A: Running Baseline evaluation.")
            baseline_metrics = evaluate_models(
                train_df, val_df, test_df, 
                exp_type='Baseline',
                dataset_name=dataset_name 
            )
            all_results.append(baseline_metrics)
            
            # Step B: TimeVAE Augmentation
            logging.info(f"[{dataset_name}] Step B: Augmenting training data.")
            augmented_train_df = augment_with_timevae(train_df)
            
            # Step C: Augmented Evaluation
            logging.info(f"[{dataset_name}] Step C: Running TimeVAE_Augmented evaluation.")
            augmented_metrics = evaluate_models(
                augmented_train_df, val_df, test_df, 
                exp_type='TimeVAE_Augmented',
                dataset_name=dataset_name
            )
            all_results.append(augmented_metrics)
            
            logging.info(f"[{dataset_name}] Pipeline completed successfully.")
            
        except Exception as e:
            logging.error(f"[{dataset_name}] Pipeline failed. Error: {e}")
            continue  # Continue to the next dataset loader
            
    # 5. Concatenate all metric DataFrames and save
    if all_results:
        final_results_df = pd.concat(all_results, ignore_index=True)
        
        # Ensure the directory exists
        os.makedirs(METRICS_DIR, exist_ok=True)
        
        output_path = os.path.join(METRICS_DIR, 'modular_experiment_results.csv')
        final_results_df.to_csv(output_path, index=False)
        logging.info(f"All experiments finished. Results securely saved to {output_path}")
    else:
        logging.warning("Pipeline finished, but no results were generated. Check dataset loaders.")

if __name__ == "__main__":
    run_experiments()