"""
all_plots.py
Generate all visualization outputs (2D/3D Pareto, evolution progress, bit heatmap).
"""

import argparse
from .utils import load_results
from .pareto_frontier import plot_pareto, plot_pareto_3d, plot_all_non_frontier, plot_pareto_loss_vs_latency
from .summary_stats import plot_summary_statistics
from .evolutionary_progress import plot_evolution_progress
from .bit_heatmap import plot_bit_allocation_heatmap


def generate_all_plots(results_dir: str, save_dir: str):
    """Run all visualization routines sequentially."""
    df = load_results(results_dir)

    plot_pareto(df, save_dir=save_dir)
    plot_pareto_loss_vs_latency(df, save_dir=save_dir)
    plot_pareto_3d(df, save_dir=save_dir)
    plot_evolution_progress(df, save_dir=save_dir)
    plot_bit_allocation_heatmap(df, save_dir=save_dir)
    plot_all_non_frontier(df, save_dir=save_dir)
    plot_summary_statistics(df, save_dir=save_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate all visualizations for quantization results.")
    parser.add_argument("--results_dir", required=True, help="Directory with JSON result files.")
    parser.add_argument("--save_dir", required=True, help="Directory to save generated plots.")
    args = parser.parse_args()

    generate_all_plots(args.results_dir, args.save_dir)


if __name__ == "__main__":
    main()
