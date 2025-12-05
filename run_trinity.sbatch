#!/bin/bash
#SBATCH --job-name=quant_trinity               # Job name  
#SBATCH -N 1 --ntasks-per-node=1          # Number of nodes and cores per node required 
#SBATCH --gres=gpu:2                                                   # Number of nodes and cores per node
#SBATCH -C 'gpu-h200'                                          # Set constraint on the type of gpu node
#SBATCH --mem-per-gpu=224GB              # Memory per CPU core, 8 CPUs/GPU 
#SBATCH -t 8:00:00                        # Duration of the job (Ex: 1 hour) 
#SBATCH -o outputs/Trinity-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences 
#SBATCH --mail-user=dshrivastava33@gatech.edu # E-mail address for notifications 

# Load modules with specific CUDA version (load separately and in correct order)
module load anaconda3
module load cuda/12.1.1
module load intel-oneapi-mkl/2023.1.0  # Intel MKL for PyTorch (fixes iJIT_NotifyEvent error)

# Initialize conda and activate environment
# Source conda.sh to initialize conda (required for conda activate)
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate trinity
else
    echo "ERROR: Could not find conda installation"
    exit 1
fi

# Set CUDA library path (helps PyTorch find CUDA libraries)
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Add Intel MKL to library path if not already there
if [ -n "$MKLROOT" ]; then
    export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH
fi

# Set HuggingFace cache paths
export HF_HOME=/home/hice1/dshrivastava33/scratch/hf_cache
export TRANSFORMERS_CACHE=/home/hice1/dshrivastava33/scratch/hf_cache
export HF_DATASETS_CACHE=/home/hice1/dshrivastava33/scratch/hf_cache

# Verify environment setup
echo "=== Environment Check ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "MKLROOT: ${MKLROOT:-not set}"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""
echo "=== PyTorch Check ==="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" 2>&1 || echo "ERROR: Could not import PyTorch - check installation and library paths"
echo ""

cd /home/hice1/dshrivastava33/Desktop/AutoLLMTuner/src/yaml_gathering

echo "[$(date)] Running hardwareyaml.py to regenerate configs/hardware.yaml"
python hardwareyaml.py

echo "[$(date)] Finished hardwareyaml.py"

# ----------------------------
#  Step 2: run evolutionary search
# ----------------------------

cd /home/hice1/dshrivastava33/Desktop/AutoLLMTuner/src

echo "[$(date)] Starting evolutionary.main_search"

echo "Beginning Trinity quantization"
python -m evolutionary.main_search \
  --config ../configs/trinity_run_config.yaml \
  --run-id trinity-mini-evoq-v001