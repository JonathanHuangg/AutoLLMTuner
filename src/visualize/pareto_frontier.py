import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .utils import load_results, ensure_dir, compute_pareto_frontier, compute_pareto_frontier_multi


def plot_pareto(df, x_col="latency_ms", y_col="accuracy",
                hue_col="seed_tag", save_dir=None):
    """Plot accuracy vs latency with Pareto frontier overlay."""
    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(8, 6))

    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=0.6, s=60)
    pareto = compute_pareto_frontier(df, x_col, y_col)
    plt.plot(pareto[x_col], pareto[y_col], "r--", linewidth=2, label="Pareto frontier")

    plt.title("Pareto Trade-off: Accuracy vs Latency")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy (1 / val_loss)")
    plt.legend()
    plt.tight_layout()

    if save_dir:
        out_dir = ensure_dir(save_dir)
        out_path = out_dir / "pareto_accuracy_vs_latency.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved 2D Pareto plot to {out_path}")
    else:
        plt.show()


def plot_pareto_3d(df, save_dir=None):
    """Plot 3D Pareto frontier (accuracy, latency, memory)."""
    objectives = {'accuracy': 'max', 'latency_ms': 'min', 'memory_gb': 'min'}
    pareto = compute_pareto_frontier_multi(df, objectives)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df["latency_ms"], df["memory_gb"], df["accuracy"],
               c='gray', alpha=0.4, label='All Points')
    ax.scatter(pareto["latency_ms"], pareto["memory_gb"], pareto["accuracy"],
               c='red', label='Pareto Frontier', s=60)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Memory (GB)")
    ax.set_zlabel("Accuracy (1 / val_loss)")
    ax.legend()
    plt.tight_layout()

    if save_dir:
        out_dir = ensure_dir(save_dir)
        out_path = out_dir / "pareto_accuracy_latency_memory.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved 3D Pareto plot to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Pareto frontiers from quantization results.")
    parser.add_argument("--results_dir", required=True, help="Directory with JSON result files.")
    parser.add_argument("--save_dir", default=None, help="Output directory for plots.")
    args = parser.parse_args()

    df = load_results(args.results_dir)
    plot_pareto(df, save_dir=args.save_dir)
    plot_pareto_3d(df, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
