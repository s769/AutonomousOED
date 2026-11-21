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


def load_data(filename, ranks_per_node, max_nodes=256, x_units="nodes"):
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


def add_efficiency_annotations(
    ax, x_vals, obs_time, ideal_time, offset=(0, 8), y_offset_data=None
):
    """Calculates efficiency and adds text annotations."""
    for i in range(len(x_vals)):
        eff = (ideal_time[i] / obs_time[i]) * 100.0
        if y_offset_data is not None:
            y_text = obs_time[i] + y_offset_data
            va = "bottom" if y_offset_data > 0 else "top"
            ax.annotate(
                f"{eff:.0f}%",
                (x_vals[i], obs_time[i]),
                xytext=(x_vals[i], y_text),
                textcoords="data",
                ha="center",
                va=va,
                fontsize=9,
                fontweight="bold",
                color="#333333",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0),
            )
        else:
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


# Systems from the paper: file prefix -> (legend label, MPI ranks per node, colour index, marker).
# Perlmutter: 1 rank per A100 (4 per node). Frontier: 1 rank per GCD, 2 GCDs per MI250X (8 per node).
KNOWN_SYSTEMS = {
    "pm": ("Perlmutter", 4, 0, "o"),
    "fr": ("Frontier", 8, 3, "s"),
}
EXTRA_MARKERS = ["^", "D", "v", "P", "X"]


def plot_scaling(systems, is_weak, out_filename, x_label, annotate=False):
    """systems: list of dicts with keys label, color, marker, data (x_vals, wall_time)."""
    systems = [s for s in systems if s["data"] is not None and len(s["data"]["x_vals"]) > 0]
    if not systems:
        print(f"No data for {out_filename}; skipping.")
        return

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    all_x = set()
    # In the paper the first system is annotated above its curve and the second below.
    strong_offsets = [(0, 10), (0, -15)]

    for i, system in enumerate(systems):
        x, wt = system["data"]["x_vals"], system["data"]["wall_time"]
        all_x.update(x)
        ax.plot(x, wt, marker=system["marker"], ms=9, color=system["color"], lw=3, zorder=5 + i)
        ideal = np.full_like(x, wt[0]) if is_weak else wt[0] * (x[0] / x)
        ax.plot(x, ideal, ls="--", color=system["color"], lw=2, alpha=0.8, zorder=4)
        if annotate and not is_weak:
            add_efficiency_annotations(ax, x, wt, ideal, offset=strong_offsets[i % 2])

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
        top_val = max(s["data"]["wall_time"].max() for s in systems)
        ax.set_ylim(top=top_val * 1.25)
        if annotate:
            y_pad = top_val * 0.07
            for i, system in enumerate(systems):
                x, wt = system["data"]["x_vals"], system["data"]["wall_time"]
                add_efficiency_annotations(
                    ax, x, wt, np.full_like(x, wt[0]), y_offset_data=y_pad if i % 2 == 0 else -y_pad
                )

    if all_x:
        ax.set_xticks(sorted(list(all_x)))

    custom_legend = [
        Line2D([0], [0], color=s["color"], lw=3, marker=s["marker"], ms=8, label=s["label"])
        for s in systems
    ]
    custom_legend.append(Line2D([0], [0], color="black", lw=2, linestyle="--", label="Ideal"))

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


def discover_systems(data_dir, kind):
    """Map prefix -> path for every <prefix>_<kind>_scaling.csv in data_dir."""
    suffix = f"_{kind}_scaling.csv"
    found = {}
    for name in sorted(os.listdir(data_dir)):
        if name.endswith(suffix) and len(name) > len(suffix):
            found[name[: -len(suffix)]] = os.path.join(data_dir, name)
    return found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot strong/weak scaling from the CSVs written by run_scaling.sh. "
            "Every <prefix>_strong_scaling.csv / <prefix>_weak_scaling.csv in --data_dir is plotted; "
            "one system is enough."
        )
    )
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--x_units", choices=["nodes", "gpus"], default="gpus")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory with the scaling CSVs (default: current directory, else ./scaling_results).",
    )
    parser.add_argument("--outdir", type=str, default=None, help="Directory for the PDFs (default: --data_dir).")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        metavar="PREFIX[=LABEL]",
        help=(
            "File prefixes to plot, optionally with a legend label, e.g. --systems fr=Frontier. "
            "Default: every prefix found. pm and fr are labelled Perlmutter and Frontier."
        ),
    )
    parser.add_argument(
        "--ranks_per_node",
        type=int,
        default=None,
        help=(
            "MPI ranks per node for prefixes other than pm (4) and fr (8); "
            "also overrides those if given. Default for other prefixes: 4."
        ),
    )
    parser.add_argument("--max_nodes", type=int, default=256, help="Largest node count to plot.")
    args = parser.parse_args()

    x_label = "Number of Compute Nodes" if args.x_units == "nodes" else "Number of GPUs"

    data_dir = args.data_dir
    if data_dir is None:
        here = discover_systems(".", "strong") or discover_systems(".", "weak")
        data_dir = "." if here or not os.path.isdir("scaling_results") else "scaling_results"
    outdir = args.outdir or data_dir
    os.makedirs(outdir, exist_ok=True)

    strong_files = discover_systems(data_dir, "strong")
    weak_files = discover_systems(data_dir, "weak")
    prefixes = sorted(set(strong_files) | set(weak_files), key=lambda p: (p not in KNOWN_SYSTEMS, list(KNOWN_SYSTEMS).index(p) if p in KNOWN_SYSTEMS else p))
    labels = {}
    if args.systems:
        wanted = []
        for item in args.systems:
            prefix, _, label = item.partition("=")
            wanted.append(prefix)
            if label:
                labels[prefix] = label
        unknown = [p for p in wanted if p not in prefixes]
        if unknown:
            parser.error(f"No scaling CSV found in {data_dir} for prefix(es): {', '.join(unknown)}")
        prefixes = wanted
    if not prefixes:
        parser.error(
            f"No *_strong_scaling.csv or *_weak_scaling.csv found in {data_dir}. "
            "Use --data_dir to point at the output directory of run_scaling.sh."
        )

    cb_palette = sns.color_palette("colorblind")
    used_colors = {KNOWN_SYSTEMS[p][2] for p in prefixes if p in KNOWN_SYSTEMS}
    free_colors = [i for i in range(len(cb_palette)) if i not in used_colors]
    strong, weak = [], []
    n_extra = 0
    for prefix in prefixes:
        if prefix in KNOWN_SYSTEMS:
            label, rpn, color_idx, marker = KNOWN_SYSTEMS[prefix]
        else:
            label, rpn = prefix, 4
            color_idx = free_colors[n_extra % len(free_colors)]
            marker = EXTRA_MARKERS[n_extra % len(EXTRA_MARKERS)]
            n_extra += 1
        label = labels.get(prefix, label)
        if args.ranks_per_node is not None:
            rpn = args.ranks_per_node
        style = {"label": label, "color": cb_palette[color_idx], "marker": marker}
        for files, target in ((strong_files, strong), (weak_files, weak)):
            data = (
                load_data(files[prefix], rpn, max_nodes=args.max_nodes, x_units=args.x_units)
                if prefix in files
                else None
            )
            target.append(dict(style, data=data))

    plot_scaling(strong, False, os.path.join(outdir, "strong_scaling_walltime.pdf"), x_label, args.annotate)
    plot_scaling(weak, True, os.path.join(outdir, "weak_scaling_walltime.pdf"), x_label, args.annotate)
