import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from .utils import load_results, ensure_dir, compute_pareto_frontier, compute_pareto_frontier_multi


def plot_pareto(df, x_col="latency_ms", y_col="val_loss",
                hue_col="seed_tag", save_dir=None):
    """Plot Cross Entropy Loss vs latency with Pareto frontier overlay."""
    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(8, 6))

    pareto = compute_pareto_frontier(df, x_col, y_col, objective_y="min")
    print(f"[DEBUG] Plotting 2D Pareto: total points={len(df)}, pareto points={len(pareto)}")
    

    # Plot all points first (non-Pareto points)
    plt.scatter(df[x_col], df[y_col], c='gray', alpha=0.4, s=60, label='All Points')
    
    # Overlay Pareto frontier points in red
    plt.scatter(pareto[x_col], pareto[y_col], c='red', s=100, label='Pareto Frontier', zorder=5)

    plt.title("Pareto Trade-off: Cross Entropy Loss vs Latency")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Cross Entropy Loss (val_loss)")
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

def plot_all_non_frontier(df, x_col="latency_ms", y_col="val_loss",
                          save_dir=None):
    """Plot ONLY non-Pareto points (dominated points) to inspect the cloud."""
    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(8, 6))

    # Compute frontier to know what to exclude
    pareto = compute_pareto_frontier(df, x_col, y_col, objective_y="min")
    mask_frontier = df.index.isin(pareto.index)

    n_total = len(df)
    n_frontier = mask_frontier.sum()
    n_non = (~mask_frontier).sum()
    print(f"[DEBUG] total={n_total}, frontier={n_frontier}, non_frontier={n_non}")

    # Scatter only non-frontier points
    plt.scatter(
        df.loc[~mask_frontier, x_col],
        df.loc[~mask_frontier, y_col],
        c="gray",
        alpha=0.8,
        s=40,
        label="Non-frontier points",
    )

    plt.title("All Non-frontier Points: Cross Entropy Loss vs Latency")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Cross Entropy Loss (val_loss)")
    plt.legend()
    plt.tight_layout()

    if save_dir:
        out_dir = ensure_dir(save_dir)
        out_path = out_dir / "all_non_frontier_accuracy_vs_latency.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved non-frontier plot to {out_path}")
    else:
        plt.show()
        
def plot_pareto_3d(df, save_dir=None):
    """Plot 3D Pareto frontier (val_loss, latency, memory)."""
    objectives = {'val_loss': 'min', 'latency_ms': 'min', 'memory_gb': 'min'}
    pareto = compute_pareto_frontier_multi(df, objectives)
    print(f"[DEBUG] Plotting 3D Pareto: total points={len(df)}, pareto points={len(pareto)}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points first with better visibility
    ax.scatter(df["latency_ms"], df["memory_gb"], df["val_loss"],
               c='gray', alpha=0.5, s=30, label='All Points', edgecolors='none')
    
    # Overlay Pareto frontier points
    ax.scatter(pareto["latency_ms"], pareto["memory_gb"], pareto["val_loss"],
               c='red', s=100, label='Pareto Frontier', edgecolors='darkred', linewidth=1.5, zorder=5)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Memory (GB)")
    ax.set_zlabel("Cross Entropy Loss (val_loss)")
    ax.legend(loc='upper left')
    
    # Improve 3D visualization
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

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
