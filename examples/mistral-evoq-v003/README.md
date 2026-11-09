# Mistral-7B-Instruct run (raw data)

The raw data behind `results/mistral-v003`: the per-epoch Pareto archive
snapshots (`frontier_*.json`) and the SQLite cache of every evaluation
(`metrics.sqlite`) from the `mistral-evoq-v003` run on a Georgia Tech PACE
GPU node.

## Regenerate the figures

```bash
autollmtuner-plots --results_dir examples/mistral-evoq-v003 --save_dir /tmp/mistral-v003
```

That reproduces `results/mistral-v003/` (figures + `summary_stats.json`).

## What the data says

93 unique configurations evaluated, 11 on the Pareto frontier
(1 failed evaluation, the `val_loss = 10.0` sentinel, is excluded):

| | All | Frontier |
|---|---|---|
| Cross-entropy loss (MATH-500) | 2.516 ± 0.079 | 2.478 ± 0.007 |
| Memory | 8.89 ± 4.82 GB | 8.88 ± 4.86 GB |
| Latency | 26.62 ± 11.67 ms | 26.73 ± 7.95 ms |

Frontier bit mix: 56.3% 4-bit, 6.0% 8-bit, 37.7% 16-bit. Smallest frontier
config is 4.09 GB; best loss is 2.469.

## Why this replaces the previously committed summary

The summary previously committed for this run (164 configurations,
52 frontier points) was produced by the old analysis code, which had two
bugs that are fixed in this revamp: it counted each individual twice when
it appeared in both the frontier JSONs and the SQLite cache, and it never
matched the SQLite `latency` column to `latency_ms`, so cached rows
carried NaN latency and could not be dominated out of the frontier.
Re-analyzing the same raw data with the fixed loader gives 93 unique
individuals and the 11-point frontier above. The old numbers are wrong;
these are the corrected ones.
