# Smoke run (CPU, mock backend)

Output of the CPU smoke test — not a model result.

Produced by:

```bash
autollmtuner-search --config configs/smoke_run_config.yaml --run-id smoke-cpu-mock
autollmtuner-plots --results_dir runs/smoke-cpu-mock --save_dir runs/smoke-cpu-mock/plots
```

The search runs with `evaluation.mock_backend: true`, so no model is loaded
and each individual's loss is a deterministic synthetic surrogate
(`evolutionary.evaluator.surrogate_mock_loss`) that rewards keeping more
bits. Latency is disabled (`measure_latency: none`) because the committed
hardware LUT has no entries for the Qwen2.5 op shapes. The point of this
run is to prove the pipeline works end to end on a machine with no GPU:
config loading, population, NSGA-II generations, the SQLite cache, frontier
saving, and the full plot pipeline.

What it produced: 18 unique evaluations over 3 generations (population 8),
an 11-point Pareto frontier with surrogate loss 0.42–0.84 and memory
5.01–12.42 GB. Re-running the commands above reproduces it exactly
(seed 67).
