import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

# ==========================================
# 1. 路径与配置 (确保与 main 脚本对齐)
# ==========================================
METRICS_PATH = "results/metrics/baseline_results.csv"  
PREDS_DIR = "results/predictions"
OUTPUT_FIG_DIR = "results/figures"

os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

# 统一模型配色，保证所有图表颜色一致
MODELS = ["IForest", "COPOD", "ECOD", "OCSVM", "LOF"]
MODEL_COLORS = dict(zip(MODELS, sns.color_palette("husl", len(MODELS))))

sns.set_theme(style="whitegrid", context="talk")

# ==========================================
# 2. 通用条形图绘制函数 (ROC & PR)
# ==========================================
def draw_overall_barplot(df, metric_name, filename):
    """绘制所有数据集下各模型的指标对比条形图"""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Dataset', y=metric_name, hue='Model', palette=MODEL_COLORS)
    
    plt.title(f'Overall Performance: {metric_name}', fontweight='bold', pad=20)
    plt.ylabel(f'{metric_name} Score')
    plt.ylim(0, 1.05)
    
    # ROC 图加一条 0.5 的随机基准线
    if 'ROC' in metric_name:
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Random Guess')
        
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_FIG_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Bar chart saved: {save_path}")

# ==========================================
# 3. 数据集详细曲线函数 (1x2 Subplots)
# ==========================================
def draw_detailed_curves(dataset_name):
    """为单个数据集绘制 ROC 和 PR 曲线对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    anomaly_rate = 0
    found_any = False

    for model_name in MODELS:
        npz_path = os.path.join(PREDS_DIR, f"{dataset_name}_{model_name}_preds.npz")
        if not os.path.exists(npz_path): continue
        
        found_any = True
        data = np.load(npz_path)
        y_test, y_scores = data['y_test'], data['y_scores']
        anomaly_rate = np.mean(y_test)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        ax1.plot(fpr, tpr, label=f'{model_name} ({auc(fpr, tpr):.3f})', color=MODEL_COLORS[model_name], lw=2)
        
        # PR
        prec, rec, _ = precision_recall_curve(y_test, y_scores)
        ax2.plot(rec, prec, label=f'{model_name} ({average_precision_score(y_test, y_scores):.3f})', color=MODEL_COLORS[model_name], lw=2)

    if not found_any: return

    # 装饰 ROC 子图
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax1.set_title(f'{dataset_name}: ROC Curves')
    ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
    ax1.legend(fontsize=10)

    # 装饰 PR 子图
    ax2.axhline(y=anomaly_rate, color='k', linestyle='--', alpha=0.4, label=f'Random ({anomaly_rate:.2%})')
    ax2.set_title(f'{dataset_name}: PR Curves')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_FIG_DIR, f"{dataset_name}_detailed_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Detailed curves saved: {save_path}")

# ==========================================
# 4. 执行
# ==========================================
if __name__ == "__main__":
    if os.path.exists(METRICS_PATH):
        df = pd.read_csv(METRICS_PATH)
        
        # 1. 绘制全局 ROC 条形图
        draw_overall_barplot(df, 'Test_ROC-AUC', 'overall_roc_barplot.png')
        
        # 2. 【找回来了！】绘制全局 PR 条形图
        draw_overall_barplot(df, 'Test_PR-AUC', 'overall_pr_barplot.png')
        
        # 3. 绘制每个数据集的详细对比图
        for ds in df['Dataset'].unique():
            draw_detailed_curves(ds)
            
        print("\n🎉 所有图表已生成在 results/figures 目录下！")
    else:
        print(f"❌ 错误：找不到文件 {METRICS_PATH}，请先运行 baseline 脚本。")