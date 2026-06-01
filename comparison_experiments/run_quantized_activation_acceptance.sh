#!/bin/bash
#SBATCH -J speq_accept
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
QUANTIZED_MODEL_DIR="/data2/group_谈海生/lagin/models/Qwen3-30B-A3B-Base-experts-int4-g128"

if [ ! -f "${QUANTIZED_MODEL_DIR}/quantization_config.json" ]; then
  echo "Missing quantized draft model: ${QUANTIZED_MODEL_DIR}"
  echo "Submit first: sbatch comparison_experiments/run_quantize_qwen3_experts.sh"
  exit 1
fi

srun .venv/bin/python -u comparison_experiments/quantized_activation_acceptance.py \
  --quantized-model-dir "${QUANTIZED_MODEL_DIR}" \
  --quantized-weight-device cpu \
  --cache-rates 0.25,0.5,0.75,1.0 \
  --draft-lengths 1,2,4,8 \
  --max-samples 20 \
  --max-new-tokens 32 \
  --max-prompt-tokens 1024

echo "Job finished at: $(date)"
