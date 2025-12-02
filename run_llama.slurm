#!/bin/bash
#SBATCH -J llama-evoq
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

echo "=== Loading Modules ==="
module load anaconda3
module load cuda/12.1.1
module load gcc/11.2.0

echo "=== Activating Environment ==="
conda activate /storage/ice1/1/5/lalexeyev3/conda_envs/autollmtuner

echo "=== Navigating to Source Directory ==="
cd /storage/ice1/1/5/lalexeyev3/AutoLLMTuner-1/src

echo "=== Setting HF Cache Paths ==="
export HF_HOME=/storage/ice1/1/5/lalexeyev3/scratch/hf_cache
export TRANSFORMERS_CACHE=/storage/ice1/1/5/lalexeyev3/scratch/hf_cache
export HF_DATASETS_CACHE=/storage/ice1/1/5/lalexeyev3/scratch/hf_cache

mkdir -p logs

cd /storage/ice1/1/5/lalexeyev3/llm_quant/src/yaml_gathering

echo "=== Running Evolutionary Search on LLaMA ==="
python -m evolutionary.main_search \
  --config ../configs/llama7b_run_config.yaml \
  --run-id llama_h200_8gpu_run1

echo "=== Job Complete ==="
