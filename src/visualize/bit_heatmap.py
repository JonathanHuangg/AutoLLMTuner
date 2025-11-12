import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .utils import ensure_dir

def plot_bit_allocation_heatmap(df, save_dir=None):
    """Visualize bit allocations per layer across individuals."""
    if "bits" not in df.columns:
        print("[WARN] No 'bits' column found for heatmap.")
        return

    bit_matrix = np.stack(df["bits"].to_numpy())
    plt.figure(figsize=(10, 6))
    sns.heatmap(bit_matrix, cmap="magma", cbar_kws={'label': 'Bit Width'})
    plt.xlabel("Layer Index")
    plt.ylabel("Model Index")
    plt.title("Per-Layer Bit Allocation Heatmap")
    plt.tight_layout()

    if save_dir:
        out_path = ensure_dir(save_dir) / "bit_allocation_heatmap.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved bit allocation heatmap to {out_path}")
    else:
        plt.show()
