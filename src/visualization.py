"""
visualization.py — Generates comparative performance plots with absolute delta annotations.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dynamically append the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    import config
except ImportError:
    print("Error: Unable to import config.py. Please ensure the script is executed from the project root.")
    sys.exit(1)

def plot_pipeline_comparison_with_absolute_deltas() -> None:
    metrics_csv = os.path.join(config.METRICS_DIR, "pipeline_results.csv")
    output_dir = os.path.join(config.RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(metrics_csv):
        print(f"Error: Metrics file not found at {metrics_csv}")
        return

    df = pd.read_csv(metrics_csv)
    df_test = df[df['split'] == 'test'].copy() if 'split' in df.columns else df.copy()
        
    datasets = df_test['dataset'].unique()
    sns.set_theme(style="whitegrid")
    
    for dataset in datasets:
        subset = df_test[df_test['dataset'] == dataset]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Test Set Performance with Absolute Delta (Δ): {dataset}', fontsize=16, fontweight='bold')
        
        metrics = [('roc_auc', 'ROC-AUC Score', axes[0]), ('pr_auc', 'PR-AUC Score', axes[1])]
        
        for metric_col, title, ax in metrics:
            sns.barplot(
                data=subset, x='model', y=metric_col, hue='pipeline', 
                ax=ax, palette=['#4C72B0', '#DD8452'], hue_order=['baseline', 'augmented']
            )
            ax.set_title(title, fontsize=14)
            ax.set_ylabel('Score')
            ax.set_xlabel('Model')
            
            # 1. Annotate the absolute numerical values on the bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', label_type='edge', padding=3, fontsize=10)
                
            max_val = subset[metric_col].max()
            ax.set_ylim(0, min(1.15, max_val * 1.45)) 
            ax.legend(title='Pipeline', loc='lower right' if metric_col == 'roc_auc' else 'upper right')
            
            # 2. Draw Absolute Delta Annotations (Δ)
            if len(ax.containers) >= 2:
                bars_base = ax.containers[0]
                bars_aug = ax.containers[1]
                
                for b_base, b_aug in zip(bars_base, bars_aug):
                    x_base = b_base.get_x() + b_base.get_width() / 2
                    y_base = b_base.get_height()
                    x_aug = b_aug.get_x() + b_aug.get_width() / 2
                    y_aug = b_aug.get_height()
                    
                    # Calculate Absolute Delta instead of Percentage Lift
                    delta = y_aug - y_base
                        
                    # Skip negligible changes (e.g., < 0.001)
                    if abs(delta) < 0.001: 
                        continue
                        
                    color = 'green' if delta > 0 else 'red'
                    sign = '+' if delta > 0 else ''
                        
                    y_max = max(y_base, y_aug)
                    y_arrow = y_max + max_val * 0.12 
                    
                    ax.annotate(
                        "",
                        xy=(x_aug, y_arrow), xycoords='data',
                        xytext=(x_base, y_arrow), textcoords='data',
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5, shrinkA=0, shrinkB=0)
                    )
                    
                    text_x = (x_base + x_aug) / 2
                    text_y = y_arrow + max_val * 0.02
                    # Display as Δ +0.031
                    ax.text(
                        text_x, text_y, 
                        f"Δ {sign}{delta:.3f}", 
                        ha='center', va='bottom', color=color, fontweight='bold', fontsize=10
                    )
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{dataset}_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Successfully generated and saved: {save_path}")

if __name__ == "__main__":
    plot_pipeline_comparison_with_absolute_deltas()