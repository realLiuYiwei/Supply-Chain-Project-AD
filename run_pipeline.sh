#!/bin/bash
#SBATCH --job-name=qc_anomaly_pipeline    # 作业名称
#SBATCH --output=logs/pipeline_%j.out     # 标准输出日志
#SBATCH --error=logs/pipeline_%j.err      # 错误输出日志
#SBATCH --partition=gpu                   # 提交到 gpu 队列 (Max Time: 3 hours)
#SBATCH --nodes=1                         # 单节点运行
#SBATCH --ntasks=1                        # 单任务
#SBATCH --cpus-per-task=8                 # 分配 8 个 CPU 核心，用于 Pandas/Sklearn 预处理
#SBATCH --mem=32G                         # 分配 32GB 内存，确保时间序列滑动窗口不 OOM
#SBATCH --gres=gpu:1                      # 申请 1 张 GPU 用于 TimeOmniVAE 加速
#SBATCH --time=03:00:00                   # 预估运行时间：3小时 (gpu 队列的上限)

# 创建日志目录（如果不存在）
mkdir -p logs

# 加载环境
CONDA_PATH=$HOME/miniconda3
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate sc_ad

# 确保 Python 能找到 src 模块
export PYTHONPATH=$PWD:$PYTHONPATH

# 运行主程序 (根据项目结构修正了入口文件)
python main.py