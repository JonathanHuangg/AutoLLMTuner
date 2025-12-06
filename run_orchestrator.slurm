#!/bin/bash
#SBATCH -J orchestrator-evoq
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

echo "=== Loading Modules ==="
module load anaconda3
module load cuda/12.6.1

echo "=== Activating Environment ==="
conda activate /home/hice1/slee3755/conda_envs/autollmtuner

echo "=== Navigating to Project Directory ==="
cd /home/hice1/slee3755/AutoLLMTuner

echo "=== Setting HF Cache Paths ==="
export HF_HOME=/home/hice1/slee3755/scratch/hf_cache
export TRANSFORMERS_CACHE=/home/hice1/slee3755/scratch/hf_cache
export HF_DATASETS_CACHE=/home/hice1/slee3755/scratch/hf_cache

mkdir -p logs

echo "=== Running Evolutionary Search (Orchestrator 8B) ==="
python -m src.evolutionary.main_search \
  --config configs/orchestrator_run_config.yaml \
  --run-id Orchestrator-evoq-v001

echo "=== Job Complete ==="
