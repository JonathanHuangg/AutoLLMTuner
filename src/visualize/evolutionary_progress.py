import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .utils import ensure_dir

def plot_evolution_progress(df, save_dir=None):
    """Plot best and average fitness per generation."""
    if "generation" not in df.columns or "fitness" not in df.columns:
        print("[WARN] No generation or fitness data available.")
        return

    sns.set(style="whitegrid", context="talk")
    progress = df.groupby("generation")["fitness"].agg(["mean", "max"]).reset_index()

    plt.figure(figsize=(7,5))
    plt.plot(progress["generation"], progress["max"], lw=2, label="Best Fitness")
    plt.plot(progress["generation"], progress["mean"], lw=2, ls="--", label="Mean Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Evolutionary Progress Over Generations")
    plt.legend()
    plt.tight_layout()

    if save_dir:
        out_path = ensure_dir(save_dir) / "evolution_progress.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved evolution progress plot to {out_path}")
    else:
        plt.show()
