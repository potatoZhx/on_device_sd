#!/bin/bash
#SBATCH -J cache_prior_accept
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -p A800
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "Running on host: $(hostname)"
echo "Starting time: $(date)"

module load cuda/12.9.1
module load Anaconda3/2025.06

cd ~/MOE_SD
source .venv/bin/activate
mkdir -p ./logs

export OMP_NUM_THREADS=4

srun .venv/bin/python -u comparison_experiments/cache_prior_acceptance.py \
  --cache-rates 0.25,0.5,0.75,1.0 \
  --draft-lengths 1,2,4,8 \
  --max-samples 20 \
  --max-new-tokens 32 \
  --max-prompt-tokens 1024 \
  --lambda-val 0.5 \
  --top-j 2

echo "Job finished at: $(date)"
