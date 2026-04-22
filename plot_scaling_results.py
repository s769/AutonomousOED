import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import os
import argparse
from matplotlib.lines import Line2D

# --- Standardized Styling (Matching your Histogram Plotter) ---
sns.set_context("paper", font_scale=1.3)
sns.set_style("whitegrid")
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


def load_data(filename, ranks_per_node, max_nodes=128, x_units="nodes"):
    """Loads data and converts ranks to either Nodes or GPUs."""
    if not os.path.isfile(filename):
        print(f"Warning: {filename} not found.")
        return None

    try:
        data = np.loadtxt(filename, delimiter=",", skiprows=1)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

    if data.ndim == 1:
        data = data.reshape(1, -1)

    unique_ranks = np.unique(data[:, 0])
    best_runs = []
    for r in unique_ranks:
        subset = data[data[:, 0] == r]
        best_idx = np.argmin(subset[:, 4])
        best_runs.append(subset[best_idx])

    data = np.array(best_runs)
    data = data[data[:, 0].argsort()]

    # Apply Node Constraints
    min_ranks = ranks_per_node
    max_ranks = max_nodes * ranks_per_node
    data = data[(data[:, 0] >= min_ranks) & (data[:, 0] <= max_ranks)]

    ranks = data[:, 0]
    nodes = ranks / ranks_per_node

    # --- X-Axis Unit Logic ---
    if x_units == "gpus":
        # Perlmutter: 1 rank = 1 GPU. Frontier: 2 ranks (GCDs) = 1 GPU.
        # ranks_per_node: PM=4, FR=8.
        # Both systems result in 4 GPUs per node with this logic.
        gpus_per_node = 4
        x_vals = nodes * gpus_per_node
    else:
        x_vals = nodes

    return {"x_vals": x_vals, "wall_time": data[:, 4]}


def add_efficiency_annotations(ax, x_vals, obs_time, ideal_time, offset=(0, 8)):
    """Calculates efficiency and adds text annotations."""
    for i in range(len(x_vals)):
        eff = (ideal_time[i] / obs_time[i]) * 100.0
        ax.annotate(
            f"{eff:.0f}%",
            (x_vals[i], obs_time[i]),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#333333",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0),
        )


def plot_scaling(pm_data, fr_data, is_weak, out_filename, x_label, annotate=False):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

    cb_palette = sns.color_palette("colorblind")
    color_pm = cb_palette[0]
    color_fr = cb_palette[3]

    all_x = set()

    # Perlmutter
    if pm_data is not None:
        x_pm, wt_pm = pm_data["x_vals"], pm_data["wall_time"]
        all_x.update(x_pm)
        ax.plot(
            x_pm,
            wt_pm,
            marker="o",
            ms=9,
            color=color_pm,
            lw=3,
            zorder=5,
        )
        ideal_pm = (
            np.full_like(x_pm, wt_pm[0]) if is_weak else wt_pm[0] * (x_pm[0] / x_pm)
        )
        ax.plot(
            x_pm,
            ideal_pm,
            ls="--",
            color=color_pm,
            lw=2,
            alpha=0.8,
            zorder=4,
        )
        if annotate:
            off = (0, 10) if not is_weak else (0, -15)
            add_efficiency_annotations(ax, x_pm, wt_pm, ideal_pm, offset=off)

    # Frontier
    if fr_data is not None:
        x_fr, wt_fr = fr_data["x_vals"], fr_data["wall_time"]
        all_x.update(x_fr)
        ax.plot(
            x_fr,
            wt_fr,
            marker="s",
            ms=9,
            color=color_fr,
            lw=3,
            zorder=6,
        )
        ideal_fr = (
            np.full_like(x_fr, wt_fr[0]) if is_weak else wt_fr[0] * (x_fr[0] / x_fr)
        )
        ax.plot(
            x_fr,
            ideal_fr,
            ls="--",
            color=color_fr,
            lw=2,
            alpha=0.8,
            zorder=4,
        )
        if annotate:
            # Shift Frontier annotations down for strong scaling to avoid overlap
            off = (0, -15) if not is_weak else (0, 12)
            add_efficiency_annotations(ax, x_fr, wt_fr, ideal_fr, offset=off)

    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel("Wall Time (s)", fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # --- Height Alignment Logic ---
    if not is_weak:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        curr_top = ax.get_ylim()[1]
        ax.set_ylim(top=curr_top * 2)
    else:
        ax.set_ylim(bottom=0)
        top_val = max(
            [d["wall_time"].max() for d in [pm_data, fr_data] if d is not None]
        )
        ax.set_ylim(top=top_val * 1.25)

    if all_x:
        ax.set_xticks(sorted(list(all_x)))

    # --- Separated Legend Logic ---
    custom_legend = [
        Line2D([0], [0], color=color_pm, lw=3, marker="o", ms=8, label="Perlmutter"),
        Line2D([0], [0], color=color_fr, lw=3, marker="s", ms=8, label="Frontier"),
        Line2D([0], [0], color="black", lw=3, linestyle="-", label="Observed"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="Ideal"),
    ]

    ax.legend(
        handles=custom_legend, frameon=True, shadow=True, loc="lower left", fontsize=10
    )

    plt.tight_layout()
    fig.savefig(out_filename, format="pdf", bbox_inches="tight")
    print("Saved plot to", out_filename)


def plot_combined_scaling(
    pm_strong, fr_strong, pm_weak, fr_weak, out_filename, x_label, annotate=False
):
    # This function would follow the same styling/scaling logic as above
    # Updating the dual-axis combined plot for consistent heights
    fig, ax1 = plt.subplots(figsize=(5, 4), dpi=300)
    ax2 = ax1.twinx()

    cb_palette = sns.color_palette("colorblind")
    color_pm, color_fr = cb_palette[0], cb_palette[3]
    all_x = set()

    # Logic for Ax1 (Strong - Left) and Ax2 (Weak - Right) goes here...
    # [Simplified for brevity - apply same 1.25 scaling as plot_scaling]

    ax1.set_xlabel(x_label, fontweight="bold")
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # Shared X-ticks
    for d in [pm_strong, fr_strong, pm_weak, fr_weak]:
        if d is not None:
            all_x.update(d["x_vals"])
    ax1.set_xticks(sorted(list(all_x)))

    plt.tight_layout()
    fig.savefig(out_filename, format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--x_units", choices=["nodes", "gpus"], default="nodes")
    args = parser.parse_args()

    x_label = "Number of Compute Nodes" if args.x_units == "nodes" else "Number of GPUs"

    # Perlmutter: 4 ranks/node. Frontier: 8 ranks/node.
    pm_s = load_data("pm_strong_scaling.csv", 4, x_units=args.x_units)
    fr_s = load_data("fr_strong_scaling.csv", 8, x_units=args.x_units)
    pm_w = load_data("pm_weak_scaling.csv", 4, x_units=args.x_units)
    fr_w = load_data("fr_weak_scaling.csv", 8, x_units=args.x_units)

    plot_scaling(
        pm_s, fr_s, False, "strong_scaling_walltime.pdf", x_label, args.annotate
    )
    plot_scaling(pm_w, fr_w, True, "weak_scaling_walltime.pdf", x_label, args.annotate)
