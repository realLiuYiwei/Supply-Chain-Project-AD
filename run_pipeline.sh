#!/bin/bash
#SBATCH --job-name=qc_anomaly_pipeline    # 作业名称
#SBATCH --output=logs/pipeline_%j.out     # 标准输出日志
#SBATCH --error=logs/pipeline_%j.err      # 错误输出日志
#SBATCH --partition=gpu                   # 提交到的队列 (NUS SoC 常用 gpu 或 xgpu，请根据权限调整)
#SBATCH --nodes=1                         # 单节点运行
#SBATCH --ntasks=1                        # 单任务
#SBATCH --cpus-per-task=8                 # 分配8个CPU核心，用于Pandas数据加载和Sklearn预处理
#SBATCH --mem=32G                         # 分配32GB内存，确保时间序列滑动窗口不OOM
#SBATCH --gres=gpu:1                      # 申请1张GPU用于 TimeOmniVAE 加速
#SBATCH --time=24:00:00                   # 预估运行时间，按需调整

# 创建日志目录（如果不存在）
mkdir -p logs

# 加载环境 (如果你使用 conda，请取消注释并修改为你的环境名)
CONDA_PATH=$HOME/miniconda3
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate sc_ad

export PYTHONPATH=$PWD:$PYTHONPATH

# 运行主程序
python -R src/run_experiments.py