import argparse
from .utils import load_results, slice_pareto

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--metrics", nargs="+", default=["val_loss", "latency_ms", "memory_gb"])

    parser.add_argument("--latency-min", type=float)
    parser.add_argument("--latency-max", type=float)
    parser.add_argument("--memory-min", type=float)
    parser.add_argument("--memory-max", type=float)
    parser.add_argument("--val-loss-min", type=float)
    parser.add_argument("--val-loss-max", type=float)

    parser.add_argument("--n-best", type=int, default=3)

    args = parser.parse_args()

    df = load_results(args.results_dir)

    result = slice_pareto(
        df,
        metrics_for_pareto=args.metrics,
        latency_min=args.latency_min,
        latency_max=args.latency_max,
        memory_min=args.memory_min,
        memory_max=args.memory_max,
        val_loss_min=args.val_loss_min,
        val_loss_max=args.val_loss_max,
        n_best=args.n_best,
    )

    print(result)

if __name__ == "__main__":
    main()

