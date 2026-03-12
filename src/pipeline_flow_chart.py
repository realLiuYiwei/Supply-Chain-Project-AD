from graphviz import Digraph

def generate_pipeline_flowchart():
    # 创建有向图
    dot = Digraph(comment='Supply-Chain Anomaly Detection Pipeline', format='png')
    dot.attr(rankdir='TB', size='12,12', fontname='Helvetica')
    dot.attr('node', shape='box', style='filled, rounded', fillcolor='lightblue', fontname='Helvetica')

    # Data Loading
    dot.node('A', 'Raw Industrial Data\n(SECOM / AI4I / Wafer)', shape='cylinder', fillcolor='lightgrey')
    dot.node('B', 'Load, Sort Chronologically\n& Drop Leaking Cols')
    dot.node('C', 'Determine Split Boundaries\n(Train 60% : Val 20% : Test 20%)')
    
    dot.edges([('A', 'B'), ('B', 'C')])

    # Baseline Pipeline Subgraph
    with dot.subgraph(name='cluster_baseline') as c:
        c.attr(label='Baseline Pipeline', style='dashed', color='blue')
        # 明确指出 Fit 仅发生在 Train 集
        c.node('D1', 'Global Preprocess (Baseline)\n Fit Scaler on Normal Train Only\n Transform Global Timeline\n Scale continuous only', fillcolor='lightyellow')
        c.node('D2', 'Sliding Window Features\n Global Timeline (T=5)', fillcolor='lightyellow')
        c.node('D3', 'Split Data & Clean Train\n(Move anomalies to Val)')
        c.node('D4', 'Train PyOD Models\n(IForest, COPOD, ECOD, LOF)', fillcolor='lightgreen')
        c.node('D5', 'Evaluate (ROC-AUC, PR-AUC)', shape='ellipse', fillcolor='lightpink')
        
        c.edges([('D1', 'D2'), ('D2', 'D3'), ('D3', 'D4'), ('D4', 'D5')])

    # VAE Pipeline Subgraph
    with dot.subgraph(name='cluster_vae') as c:
        c.attr(label='TimeOmniVAE Augmented Pipeline', style='dashed', color='red')
        # 明确 VAE 分支也包含了 Scale 且 Fit 仅在 Train 集
        c.node('V1', 'Global Preprocess (VAE)\n Fit Scaler on Normal Train Only\n NaN -> 0 (Imputation)', fillcolor='lightyellow')
        c.node('V2', 'Build VAE Windows')
        c.node('V3', 'Select Normal-Train Windows')
        c.node('V4', 'Compute Category Proportions\n Conditioned Gen', fillcolor='lightyellow')
        c.node('V5', 'Train TimeOmniVAE\n& Generate Synthetic Data', fillcolor='#ffcccb') # Highlight VAE
        c.node('V6', 'Snap Categoricals & Aggregate')
        
        c.edges([('V1', 'V2'), ('V2', 'V3'), ('V3', 'V4'), ('V4', 'V5'), ('V5', 'V6')])

    dot.edge('C', 'D1')
    dot.edge('C', 'V1')

    # Combine Node
    dot.node('Merge', 'Combine Features\n(Real Train + Synthetic Train)', shape='diamond', fillcolor='orange')
    dot.edge('D3', 'Merge', label=' Baseline Train\n Features')
    dot.edge('V6', 'Merge', label=' Augmented\n Features')

    # Final Evaluation for Augmented
    dot.node('F1', 'Train PyOD Models', fillcolor='lightgreen')
    dot.node('F2', 'Evaluate Augmented', shape='ellipse', fillcolor='lightpink')
    dot.edge('Merge', 'F1')
    # Connect Val/Test sets to Final models
    dot.edge('D3', 'F1', label=' Val/Test', style='dotted') 
    dot.edge('F1', 'F2')

    # 渲染保存，关闭 view 以防止服务器报错，开启 cleanup 清理多余文本文件
    dot.render('pipeline_flowchart', view=False, cleanup=True)
    print("Flowchart successfully generated as pipeline_flowchart.png!")

if __name__ == '__main__':
    generate_pipeline_flowchart()