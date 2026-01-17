#!/bin/bash
#SBATCH -J wiki_experiment      # 作业名称
#SBATCH --nodes=1               # 使用 1 个节点
#SBATCH --ntasks-per-node=1     # 运行 1 个任务
#SBATCH --gres=gpu:1            # 申请 1 张 GPU 卡
#SBATCH --cpus-per-task=8       # 申请 8 个 CPU 核心
#SBATCH -p A800                 # 分区名称 (请根据服务器实际情况修改)
#SBATCH -w gpu5
#SBATCH --output=logs/%x-%j.out # 标准输出日志
#SBATCH --error=logs/%x-%j.err  # 错误日志

# --- 1. 环境准备 ---
echo "Running on host: $(hostname)"
echo "Starting time: $(date)"

# 加载 CUDA 模块
module load cuda/12.9.1
module load Anaconda3/2025.06

# 激活虚拟环境
cd ~/MOE_SD
source .venv/bin/activate

mkdir -p ./logs

# 优化设置
export OMP_NUM_THREADS=4    # 限制 CPU 线程数
PYTHON_SCRIPT="./get_sd_data/wiki_experiment.py"

# --- 3. 启动命令 ---
echo "Running Expert Subset Inference Script: $PYTHON_SCRIPT"
# 使用 Python 运行，-u 参数保证日志实时输出
python -u $PYTHON_SCRIPT

echo "Job finished at: $(date)"