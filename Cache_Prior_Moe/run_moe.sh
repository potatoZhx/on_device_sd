#!/bin/bash
#SBATCH -J moe                  # 作业名称
#SBATCH --nodes=1               # 使用 1 个节点
#SBATCH --ntasks-per-node=1     # 运行 1 个任务
#SBATCH --gres=gpu:1            # 【关键修改】只申请 1 张 GPU 卡
#SBATCH --cpus-per-task=8       # 申请 8 个 CPU 核心 (推理任务不需要太多 CPU)
#SBATCH -p A800                 # 分区名称 (请根据服务器实际情况修改，如 gpu, A100, 3090 等)
#SBATCH -w gpu5
#SBATCH --output=logs/%x-%j.out # 标准输出日志 (会自动创建 logs 目录)
#SBATCH --error=logs/%x-%j.err  # 错误日志

# --- 1. 环境准备 ---
echo "Running on host: $(hostname)"
echo "Starting time: $(date)"

# 加载 CUDA 模块 (保持和你参考脚本一致，或者根据服务器情况修改)
module load cuda/12.9.1
module load Anaconda3/2025.06

# 激活你的虚拟环境
cd ~/MOE_SD
source .venv/bin/activate

mkdir -p ./logs

# 优化设置
export OMP_NUM_THREADS=4    # 限制 CPU 线程数，防止过多占用

# --- 2. 变量定义 ---
# PYTHON_SCRIPT="./Cache_Prior_Moe/run_wikitext_eval.py"
# PYTHON_SCRIPT="./Cache_Prior_Moe/run_mmlu_eval.py"
PYTHON_SCRIPT="./Cache_Prior_Moe/run_gsm8k_eval.py"

# --- 3. 启动命令 ---
echo "Running Inference Script: $PYTHON_SCRIPT"
# 直接使用 Python 运行
# unbuffered (-u) 保证日志实时输出到 .out 文件，不会缓存
python -u $PYTHON_SCRIPT

echo "Job finished at: $(date)"