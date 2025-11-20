#!/bin/bash
#SBATCH -J qwen-evoq
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# ----------------------------
#  Environment setup
# ----------------------------

module load anaconda3/2023.03
# If you need CUDA explicitly on PACE, uncomment this:
# module load cuda/12.1

# Activate your scratch-based conda env
source activate /storage/ice1/2/3/jhuang691/conda_envs/autollmtuner

# Force HuggingFace caches into scratch (avoid $HOME quota)
export HF_HOME=/storage/ice1/2/3/jhuang691/scratch/hf_cache
export TRANSFORMERS_CACHE=/storage/ice1/2/3/jhuang691/scratch/hf_cache
export HF_DATASETS_CACHE=/storage/ice1/2/3/jhuang691/scratch/hf_cache

# Make sure logs dir exists (relative to where you call sbatch)
mkdir -p logs

# ----------------------------
#  Step 1: build hardware.yaml
# ----------------------------

# Go to yaml_gathering and run the LUT generator
cd /storage/ice1/2/3/jhuang691/AutoLLMTuner/src/yaml_gathering

echo "[$(date)] Running hardwareyaml.py to regenerate configs/hardware.yaml"
python hardwareyaml.py

echo "[$(date)] Finished hardwareyaml.py"

# ----------------------------
#  Step 2: run evolutionary search
# ----------------------------

cd /storage/ice1/2/3/jhuang691/AutoLLMTuner/src

echo "[$(date)] Starting evolutionary.main_search"

python -m evolutionary.main_search \
  --config ../configs/qwen2.5_run_config.yaml \
  --run-id qwen2.5-evoq-v001

echo "[$(date)] Finished evolutionary.main_search"
