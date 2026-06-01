#!/bin/bash
#SBATCH -J qwen3_int4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -p A800
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "Running on host: $(hostname)"
echo "Starting time: $(date)"

module load cuda/12.9.1
module load Anaconda3/2025.06

cd ~/MOE_SD
source .venv/bin/activate
mkdir -p ./logs

export OMP_NUM_THREADS=4

srun .venv/bin/python -u comparison_experiments/quantize_qwen3_moe_experts.py \
  --model-path /data2/group_谈海生/lagin/models/Qwen3-30B-A3B-Base \
  --output-dir /data2/group_谈海生/lagin/models/Qwen3-30B-A3B-Base-experts-int4-g128 \
  --group-size 128 \
  --dtype bf16

echo "Job finished at: $(date)"
