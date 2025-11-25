# AutoLLMTuner

Evolutionary algorithm-based framework for optimizing mixed-precision quantization of Large Language Models using multi-objective optimization.

## Overview

AutoLLMTuner uses evolutionary algorithms (NSGA-II) to automatically discover optimal bit-width allocations for different layers of an LLM, balancing three competing objectives:
- **Model Accuracy** (validation loss on math problems)
- **Memory Footprint** (model size in bytes)
- **Inference Latency** (execution speed in milliseconds)

The system produces a Pareto frontier of solutions, allowing users to select the optimal trade-off for their specific hardware and performance requirements.

## Datasets

### MATH-500
**Source:** [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)

**Description:** A curated subset of 500 challenging mathematics problems from the MATH dataset, covering topics including algebra, geometry, probability, number theory, and calculus. Each problem includes:
- Problem statement (natural language)
- Step-by-step solution
- Final numerical or symbolic answer

**Usage in Project:** The dataset is used to evaluate the accuracy of quantized models by:
1. Loading the test split (`test.jsonl`) from HuggingFace
2. Currently evaluating on the first 50 problems for efficiency
3. Measuring model performance via a two-stage process:
   - Quantized model generates solutions
   - Full-precision checker model grades the answers (0 or 1)
4. Computing validation loss as `1 - accuracy`

**Access:** Automatically downloaded from HuggingFace Hub via Polars:
```python
pl.read_ndjson("hf://datasets/HuggingFaceH4/MATH-500/test.jsonl")
```

**Location in Code:** `src/evolutionary/bnb_quantize.py:73-81`

## Performance Measurement Tools

### 1. BitsAndBytes Quantization Library
**Repository:** [TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

**Description:** A lightweight wrapper around CUDA custom functions for 8-bit and 4-bit matrix multiplication and quantization. Provides:
- `Linear4bit`: 4-bit NormalFloat (NF4) quantization optimized for neural networks
- `Linear8bitLt`: 8-bit quantization with row-wise scaling

**Usage in Project:**
- Implements actual quantization (not simulation) for realistic accuracy measurements
- Supports dynamic layer-wise quantization with mixed precision
- Caches FP16 weights and rebuilds layers on-the-fly with different bit-widths

**Location in Code:** `src/evolutionary/bnb_quantize.py`

### 2. Hardware Latency Benchmarking Tool
**Description:** Custom GPU benchmarking system that measures real-world inference latency for different quantization schemes.

**Features:**
- Benchmarks GEMM (General Matrix Multiply) operations on target hardware
- Creates Look-Up Tables (LUT) mapping operation configurations to latency
- Supports multiple bit-widths: 2, 3, 4, 6, 8, 16 bits
- Uses PyTorch CUDA Events for microsecond-precision timing
- Warmup iterations: 10, Benchmark iterations: 50

**Measurement Process:**
1. Extracts representative matrix shapes from model architecture (attention, MLP projections)
2. Benchmarks each shape at each bit-width using real GPU operations
3. Stores results in `configs/hardware.yaml`
4. During evolution, looks up or measures new (operation, bit-width) combinations

**Location in Code:** `src/yaml_gathering/hardwareyaml.py`

### 3. Multi-Objective Fitness Evaluation
**Description:** Comprehensive evaluation system tracking three simultaneous objectives.

**Metrics:**
- **Validation Loss**: Cross-entropy loss on MATH-500 dataset (via BnbQuantizeBackend)
- **Memory Usage**: Calculated as `Σ(parameter_count × bits / 8)` bytes
- **Latency**: Sum of per-layer GEMM latencies from LUT + memory bandwidth overhead

**Evaluation Modes:**
- `LOOKUP`: Uses pre-computed LUT only (fast)
- `REAL`: Measures unknown configurations and updates LUT (accurate)

**Location in Code:** `src/evolutionary/evaluator.py`

### 4. Results Caching System
**Description:** SQLite-based cache to avoid redundant evaluations.

**Features:**
- Stores individual ID → (validation_loss, memory_bytes, latency_ms) mappings
- Persists across runs for incremental experimentation
- Significantly reduces computational cost for re-visited configurations

**Location in Code:** `src/evolutionary/cache.py`

### 5. Visualization and Analysis Tools
**Description:** Suite of plotting and statistical analysis tools.

**Components:**
- **Pareto Frontier Plots** (`pareto_frontier.py`): 2D and 3D visualizations of optimal trade-offs
- **Bit Allocation Heatmaps** (`bit_heatmap.py`): Layer-wise quantization patterns
- **Summary Statistics** (`summary_stats.py`): JSON reports with comprehensive metrics

**Libraries Used:**
- Matplotlib (≥3.8): Core plotting
- Seaborn (≥0.13): Statistical visualizations
- SciPy (≥1.11): Pareto frontier computation

**Location in Code:** `src/visualize/`

## Project Hosting and External Resources

### Model
**Name:** Qwen2.5-7B-Instruct
**URL:** [https://huggingface.co/Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
**Description:** 7-billion parameter instruction-tuned language model from Alibaba Cloud. Features:
- Architecture: Transformer decoder with 28 layers
- Hidden size: 3,584
- Intermediate size: 18,944
- Designed for chat and instruction-following tasks
- Used as the base model for quantization experiments

### Dataset
**Name:** MATH-500
**URL:** [https://huggingface.co/datasets/HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
**Description:** Subset of the competition-level MATH dataset (Hendrycks et al., 2021) containing 500 challenging problems across multiple mathematical domains. Used for evaluating reasoning capabilities of language models.

### Quantization Backend
**Name:** BitsAndBytes
**URL:** [https://github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
**Description:** 8-bit and 4-bit CUDA kernels for efficient quantization. Enables mixed-precision quantization with minimal accuracy loss through:
- NF4 (4-bit NormalFloat) quantization
- 8-bit matrix multiplication with row-wise scaling
- Dynamic quantization at runtime

### Framework and Libraries
**PyTorch:** [https://pytorch.org/](https://pytorch.org/) - Deep learning framework (≥2.3.1)
**HuggingFace Transformers:** [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers) - Model loading and inference (≥4.43.3)
**Polars:** [https://pola.rs/](https://pola.rs/) - Fast dataframe library for dataset loading
**Accelerate:** [https://huggingface.co/docs/accelerate](https://huggingface.co/docs/accelerate) - Distributed model loading

## Installation

### Requirements
- Python 3.8+
- CUDA-compatible GPU (for quantization and benchmarking)
- 16GB+ GPU memory (for Qwen2.5-7B)

### Setup
```bash
# Clone repository
git clone <repository-url>
cd AutoLLMTuner-1

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Hardware Benchmarking (Optional)
Generate a hardware-specific latency LUT:
```bash
python -m yaml_gathering.hardwareyaml
```
Output: `configs/hardware.yaml`

### 2. Run Evolutionary Search
```bash
python -m evolutionary.main_search \
  --config configs/qwen2.5_run_config.yaml \
  --run-id my_experiment
```

### 3. HPC Cluster Execution
```bash
sbatch run_qwen.slurm
```

### 4. Visualize Results
```bash
python -m visualize.pareto_frontier --results-dir results/<run-id>
python -m visualize.bit_heatmap --results-dir results/<run-id>
python -m visualize.summary_stats --results-dir results/<run-id>
```

## Configuration

Key configuration files in `configs/`:
- `qwen2.5_run_config.yaml`: Main experiment settings (population, generations, mutation rate)
- `quant_units.yaml`: Quantizable units and grouping constraints
- `qwen2.5_layer_spec.yaml`: Layer operation specifications for latency estimation
- `qwen2.5_param_counts.json`: Parameter counts per layer
- `hardware.yaml`: GPU-specific latency measurements (auto-generated)

## Experimental Results and Analysis

### Key Experiment Runs

#### 1. Correct Answers Run (Primary Results)
**Purpose:** Final production run with validated evaluation metrics
**Configurations Evaluated:** 175 unique quantization strategies
**Pareto Frontier Size:** 37 optimal configurations (21.14% of all evaluated)

**Performance Metrics:**

| Metric | All Individuals | Pareto Frontier Only |
|--------|----------------|---------------------|
| **Accuracy** (1 - val_loss) | 76.97% ± 4.79% | 79.89% ± 6.31% |
| **Memory** | 6.08 ± 1.33 GB | 5.13 ± 0.22 GB |
| **Latency** | 8.51 ± 0.07 ms | 8.49 ± 0.02 ms |

**Key Findings:**
- **Best Accuracy:** 94.0% on MATH-500 (50 problems) at 5.51 GB memory
- **Smallest Memory:** 4.83 GB while maintaining 74.0% accuracy
- **Memory Reduction:** Frontier solutions achieve **58.9% smaller** average memory compared to baseline 12.49 GB (worst case)
- **Minimal Latency Variance:** All configurations cluster tightly (8.46-8.53 ms), indicating latency is dominated by model architecture rather than bit-width

**Bit Allocation Strategy:**
- **All Individuals:** 41.90% 4-bit, 42.88% 8-bit, 15.23% 16-bit (balanced distribution)
- **Frontier Solutions:** 67.50% 4-bit, 23.77% 8-bit, 8.73% 16-bit (aggressive quantization)
- **Insight:** Optimal solutions heavily favor 4-bit quantization, suggesting most layers tolerate aggressive compression

#### 2. Overnight Everything Run (Large-Scale Validation)
**Purpose:** Comprehensive search over larger population and generations
**Configurations Evaluated:** 646 unique quantization strategies
**Pareto Frontier Size:** 81 optimal configurations (12.54% of all evaluated)

**Performance Metrics:**

| Metric | All Individuals | Pareto Frontier Only |
|--------|----------------|---------------------|
| **Accuracy** (1 - val_loss) | 42.07% ± 5.28% | 46.40% ± 4.86% |
| **Memory** | 7.08 ± 3.01 GB | 8.32 ± 3.35 GB |
| **Latency** | 8.87 ± 0.32 ms | 8.87 ± 0.32 ms |

**Observations:**
- **Lower Accuracy:** Average accuracy significantly lower (42% vs 77% in correct_answers), indicating this run may have used a different evaluation protocol or dataset sample
- **Higher Memory Variance:** Much wider distribution (std=3.01 GB vs 1.33 GB), suggesting more aggressive exploration of search space
- **Larger Frontier:** 81 optimal points vs 37, indicating more diverse trade-off solutions

**Bit Allocation Strategy:**
- **All Individuals:** 55.63% 4-bit, 25.14% 8-bit, 19.23% 16-bit
- **Frontier Solutions:** 41.70% 4-bit, 34.88% 8-bit, 23.41% 16-bit
- **Contrast:** Frontier uses more higher-precision bits (58.29% at 8/16-bit vs 41.90% in correct_answers), suggesting different accuracy requirements

### Visualization Outputs

Each experimental run generates comprehensive visualizations for analysis:

#### 1. Pareto Frontier Plots
**Files:** `pareto_accuracy_vs_latency.png`, `pareto_accuracy_latency_memory.png`

**Purpose:** Visualize the trade-off space between competing objectives
- **2D Plots:** Show accuracy vs. latency and accuracy vs. memory trade-offs
- **3D Plots:** Display all three objectives simultaneously (accuracy, latency, memory)
- **Red Points:** Pareto-optimal solutions (non-dominated configurations)
- **Blue Points:** Sub-optimal solutions (dominated by at least one other configuration)

**Key Insight:** Clear Pareto frontier demonstrates successful multi-objective optimization. Users can select any frontier point based on their specific constraints.

#### 2. Bit Allocation Heatmaps
**File:** `bit_allocation_heatmap.png`

**Purpose:** Show layer-wise quantization patterns across the 28-layer Qwen2.5 model
- **Rows:** Different configurations (individuals) from the population
- **Columns:** Model layers (0-27) and projection types (Q, K, V, O, gate, up, down)
- **Colors:** Bit-widths (4-bit = blue, 8-bit = green, 16-bit = yellow/red)

**Key Insight:** Optimal configurations show structured patterns:
- Early layers (0-5) often use more aggressive quantization (4-bit)
- Middle layers (7-15) show mixed precision strategies
- Attention projections (Q/K/V) typically share the same bit-width (due to grouping constraints)

#### 3. Summary Visualizations
**Files:** `summary_objectives.png`, `summary_bit_hist.png`, `summary_layer_mean_bits.png`

**Purpose:** Aggregate statistics across all evaluated configurations

- **Objectives Plot:** Box plots comparing accuracy, memory, and latency distributions
- **Bit Distribution Histogram:** Shows frequency of 4-bit, 8-bit, and 16-bit allocations
- **Layer-Wise Mean Bits:** Average bit-width per layer across all frontier solutions

**Key Insight:** Reveals which layers are consistently quantized to lower precision, indicating robustness to compression.

#### 4. All Non-Frontier Plot
**File:** `all_non_frontier_accuracy_vs_latency.png`

**Purpose:** Visualize the entire explored solution space, including suboptimal points
- Shows density of evaluated configurations
- Helps identify regions of the search space that were thoroughly explored
- Validates that the evolutionary algorithm effectively covered the solution space

### Comparative Analysis: Evolution Effectiveness

**Search Space Coverage:**
- **Total quantizable units:** 196 (28 layers × 7 projections per layer)
- **Effective search space:** ~70 independent decisions (due to grouping constraints)
- **Theoretical combinations:** 3^70 ≈ 4.8 × 10^33 (intractable for exhaustive search)
- **Evaluated:** 175-646 configurations (0.000...% of search space)
- **Frontier found:** 37-81 optimal solutions

**Efficiency Metrics:**
- **Frontier Ratio:** 12.54%-21.14% of evaluated configurations are Pareto-optimal
- **Implication:** High frontier ratio indicates efficient search, avoiding evaluation of obviously poor configurations

**Convergence Quality:**
- **Memory Improvement:** Up to 61.3% reduction (12.49 GB → 4.83 GB) with acceptable accuracy loss (94% → 74%)
- **Accuracy Preservation:** Best configurations maintain 94% accuracy (only 6% below FP16 baseline assumption)

### Implications

**For Deployment:**
1. **Memory-Constrained Devices:** Use 4.83 GB configuration (74% accuracy) for edge deployment
2. **Balanced Performance:** Use 5.13 GB configuration (79.89% average accuracy) for production
3. **High-Accuracy Requirements:** Use 5.51 GB configuration (94% accuracy) for critical applications

**Trade-off Analysis:**
- **20% memory reduction** (6.08 GB → 4.83 GB) costs **6% accuracy** (80% → 74%)
- **Latency is nearly constant** across all configurations, making memory-accuracy the primary trade-off

**Bit-Width Strategy:**
- Aggressive 4-bit quantization is viable for 67.5% of layers in optimal solutions
- Only 8.73% of layers require FP16 precision in best configurations
- 8-bit serves as effective middle ground for 23.77% of layers

### Artifacts Created

Each run produces:
- **JSON Statistics:** `summary_stats.json` (190-370 KB) with complete numerical analysis
  - Per-objective statistics (min, max, mean, std, quantiles)
  - Per-layer bit allocation distributions
  - Frontier vs. non-frontier comparisons
  - Accuracy bucket analysis
  - Complete frontier point data

- **PNG Visualizations:** 3-7 plots per run (40-617 KB each)
  - Publication-ready figures with clear legends and axis labels
  - High-resolution outputs suitable for presentations

- **Reproducibility:** All results tied to specific git commits and random seeds for experiment replication

### Code-Results Integration

The analysis pipeline connects code to results through:

1. **Evaluation** (`src/evolutionary/evaluator.py`):
   - Generates (val_loss, mem_bytes, latency_ms) tuples
   - Stores in SQLite cache for efficiency

2. **Frontier Computation** (`src/evolutionary/frontier.py`):
   - Implements Pareto dominance checking
   - Updates global frontier after each generation
   - Saves to JSON files

3. **Visualization** (`src/visualize/*.py`):
   - Reads frontier JSON files
   - Generates all plots automatically
   - Creates `summary_stats.json` with comprehensive analysis

4. **Results Organization:**
   - Each run creates timestamped directory
   - Preserves full evolutionary history
   - Enables comparative analysis across experiments

