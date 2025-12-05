import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .utils import ensure_dir
from sklearn.cluster import KMeans


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .utils import ensure_dir

def plot_bit_allocation_heatmap(
    df,
    save_dir=None,
    max_rows: int = 200,   # cap on number of individuals to show
):
    """
    Visualize bit allocations per layer across individuals.

    - Handles variable-length 'bits' entries by keeping only the
      most common length (assumed to be the real model layout).
    - Downsamples rows if there are too many to be informative.
    - Sorts rows by mean bit-width so patterns are visible.
    """

    if "bits" not in df.columns:
        print("[WARN] No 'bits' column found for heatmap.")
        return

    # --- 1. Collect bits and handle variable lengths ---
    bits_series = df["bits"].dropna()

    if bits_series.empty:
        print("[WARN] No non-NaN 'bits' entries for heatmap.")
        return

    lengths = np.array([len(b) for b in bits_series])
    unique_lengths, counts = np.unique(lengths, return_counts=True)
    target_len = unique_lengths[np.argmax(counts)]

    if len(unique_lengths) > 1:
        print(
            f"[INFO] Multiple bit vector lengths found {dict(zip(unique_lengths, counts))}; "
            f"using most common length = {target_len}"
        )

    mask = lengths == target_len
    filtered_bits = bits_series[mask]

    if filtered_bits.empty:
        print("[WARN] After length filtering, no rows remain for heatmap.")
        return

    # Filter: keep only lists of correct length
    clean_bits = [
        b for b in filtered_bits.to_numpy()
        if isinstance(b, (list, np.ndarray)) and len(b) == target_length
    ]

    if len(clean_bits) == 0:
        raise ValueError("No valid bit vectors of expected length found.")

    bit_matrix = np.stack(clean_bits)
    n_ind, n_layers = bit_matrix.shape

    print(f"[INFO] Heatmap using {n_ind} individuals × {n_layers} layers")

    # --- 2. Optional downsampling along individuals (rows) ---
    if n_ind > max_rows:
        step = int(np.ceil(n_ind / max_rows))
        bit_matrix = bit_matrix[::step]
        n_ind = bit_matrix.shape[0]
        print(
            f"[INFO] Downsampled rows for visualization: every {step}th individual "
            f"→ {n_ind} rows"
        )

    # --- 3. Sort rows by mean bit-width to reveal structure ---
    row_means = bit_matrix.mean(axis=1)
    order = np.argsort(row_means)
    bit_matrix = bit_matrix[order]

    # --- 4. Plot ---
    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(
        bit_matrix,
        cmap="mako",
        cbar_kws={"label": "Bit Width"},
        rasterized=True,
        vmin=bit_matrix.min(),
        vmax=bit_matrix.max(),
    )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Model Index (sorted by mean bit)")
    ax.set_title("Per-Layer Bit Allocation Heatmap")

    # Make x-axis less cluttered if many layers
    if n_layers > 40:
        tick_count = min(20, n_layers)
        xticks = np.linspace(0, n_layers - 1, num=tick_count, dtype=int)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=45, ha="right")

    plt.tight_layout()

    if save_dir:
        out_path = ensure_dir(save_dir) / "bit_allocation_heatmap.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved bit allocation heatmap to {out_path}")
    else:
        plt.show()
