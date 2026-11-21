import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import seaborn as sns
import os
import shutil
import sys
import argparse

# ==========================================
# 1. INPUT FILES
# ==========================================
# Usage:
#   python plot_single_gpu.py                      # every single_gpu_results_<gpu>.csv in ./scaling_results
#   python plot_single_gpu.py results.csv          # one CSV written by single_gpu_benchmark.py
#   python plot_single_gpu.py a.csv b.csv --labels MI250X A100
#   python plot_single_gpu.py --data_dir my_runs --outdir figures
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "scaling_results"
RESULT_PATTERN = "single_gpu_results_*.csv"

# Legend order and colours of the GPUs in the paper; any other label gets the next free colour.
KNOWN_GPUS = ["MI250X", "A100", "GH200", "GB200", "MI300X"]
KNOWN_COLOR_INDEX = {"MI250X": 1, "A100": 0, "GH200": 2, "GB200": 3, "MI300X": 4}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot single-GPU time per iteration (naive vs. Schur formulation)."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help=(
            "CSV files written by single_gpu_benchmark.py. If none are given, every "
            f"{RESULT_PATTERN} in --data_dir is plotted."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend label for each file, in order (default: taken from the file name).",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help=f"Directory searched for {RESULT_PATTERN} when no files are given (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory for the two PDFs (default: the directory of the first input file).",
    )
    parser.add_argument(
        "--no_tex",
        action="store_true",
        help="Do not use LaTeX for the labels even if it is installed.",
    )
    return parser.parse_args()


def label_from_filename(path):
    """single_gpu_results_mi250x.csv -> MI250X; anything else -> the file stem."""
    stem = Path(path).stem
    prefix = RESULT_PATTERN.split("*")[0]
    return stem[len(prefix):].upper() if stem.startswith(prefix) and len(stem) > len(prefix) else stem


def resolve_inputs(args):
    """Return a list of (path, label), in legend order."""
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        data_dir = args.data_dir or DEFAULT_DATA_DIR
        files = sorted(data_dir.glob(RESULT_PATTERN))
        if not files:
            sys.exit(
                f"No {RESULT_PATTERN} found in {data_dir}. Pass the CSV written by "
                "single_gpu_benchmark.py, e.g.: python plot_single_gpu.py results.csv --labels MI250X"
            )
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        sys.exit("File(s) not found: " + ", ".join(missing))
    if args.labels is not None and len(args.labels) != len(files):
        sys.exit(f"--labels needs one label per file ({len(files)} file(s), {len(args.labels)} label(s)).")
    labels = args.labels or [label_from_filename(f) for f in files]
    pairs = list(zip(files, labels))
    if not args.files:  # auto-discovered: keep the paper's legend order
        pairs.sort(key=lambda p: KNOWN_GPUS.index(p[1]) if p[1] in KNOWN_GPUS else len(KNOWN_GPUS))
    return pairs


ARGS = parse_args()
INPUTS = resolve_inputs(ARGS)
OUT_DIR = ARGS.outdir or INPUTS[0][0].resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_RESERVE_FRAC = 0.08
EXTEND_X_FRAC = 0.20

# ==========================================
# 2. PLOT STYLING & SETUP
# ==========================================
sns.set_context("paper", font_scale=1.3)
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")


def _ensure_latex_on_path():
    if shutil.which("latex"):
        return
    miktex_bin = Path.home() / r"AppData\Local\Programs\MiKTeX\miktex\bin\x64"
    if (miktex_bin / "latex.exe").exists():
        os.environ["PATH"] = str(miktex_bin) + os.pathsep + os.environ.get("PATH", "")


_ensure_latex_on_path()
USE_TEX = shutil.which("latex") is not None and not ARGS.no_tex

plt.rcParams["text.usetex"] = False
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.default"] = "it"
if USE_TEX:
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{bm}"

cb = sns.color_palette("colorblind")
c_theory = "#666666"


def assign_colors(labels):
    used = {KNOWN_COLOR_INDEX[l] for l in labels if l in KNOWN_COLOR_INDEX}
    free = [i for i in range(len(cb)) if i not in used]
    colors = []
    for label in labels:
        if label in KNOWN_COLOR_INDEX:
            colors.append(cb[KNOWN_COLOR_INDEX[label]])
        else:
            colors.append(cb[free.pop(0) if free else len(colors) % len(cb)])
    return colors


fig_size = (5.0, 4.0)


def load_data(filename):
    df = pd.read_csv(filename)
    needed = {"k", "time_N_IP", "time_S_IP"}
    if not needed.issubset(df.columns):
        sys.exit(f"{filename} is missing column(s): {sorted(needed - set(df.columns))}")
    return df


_colors = assign_colors([label for _, label in INPUTS])
architectures = [
    (load_data(path), color, label) for (path, label), color in zip(INPUTS, _colors)
]


def collect_finite_points(dfs, time_col):
    ks, ts = [], []
    for df in dfs:
        if df.empty:
            continue
        t = df[time_col].to_numpy(dtype=float)
        k = df["k"].to_numpy(dtype=float)
        mask = np.isfinite(t) & np.isfinite(k) & (t > 0.0) & (k > 0.0)
        ks.extend(k[mask])
        ts.extend(t[mask])
    return np.asarray(ks, dtype=float), np.asarray(ts, dtype=float)


def legend_label(name):
    return name


def cm_bold_k_mathtext():
    return r"$\mathbf{(}\boldsymbol{k}\mathbf{)}$"


def cm_bold_k_tex():
    return r"$\bm{(k)}$"


def cm_bold_big_o(power):
    if USE_TEX:
        return rf"$\bm{{\mathcal{{O}}(k^{power})}}$"
    return rf"$\mathcal{{O}}(\boldsymbol{{k}}^\mathbf{{{power}}})$"


def set_mixed_xlabel(ax, prefix="Selected Sensors"):
    """Native-font text plus TeX math, so $(k)$ matches the legend."""
    if not USE_TEX:
        ax.set_xlabel(rf"{prefix} {cm_bold_k_mathtext()}", usetex=False)
        return

    ax.set_xlabel(rf"{prefix} {cm_bold_k_mathtext()}", usetex=False)
    dummy = ax.xaxis.label
    dummy.set_alpha(0.0)
    text_props = {
        "fontsize": dummy.get_size(),
        "fontweight": "bold",
        "color": dummy.get_color(),
        "annotation_clip": False,
        "ha": "left",
        "va": "center",
    }
    prefix_artist = ax.annotate(
        prefix + " ",
        xy=(0.0, 0.5),
        xycoords=dummy,
        usetex=False,
        **text_props,
    )
    ax.annotate(
        cm_bold_k_tex(),
        xy=(1.0, 0.5),
        xycoords=prefix_artist,
        usetex=True,
        **text_props,
    )


def finite_kt(df, time_col):
    k = df["k"].to_numpy(dtype=float)
    t = df[time_col].to_numpy(dtype=float)
    mask = np.isfinite(k) & np.isfinite(t) & (k > 0.0) & (t > 0.0)
    return k[mask], t[mask]


def last_finite_point(df, time_col):
    k_valid, t_valid = finite_kt(df, time_col)
    if k_valid.size == 0:
        return None
    idx = np.argmax(k_valid)
    return float(k_valid[idx]), float(t_valid[idx])


def asymptotic_extension(k_last, t_last, power, x_max, n=40):
    if k_last <= 0.0 or t_last <= 0.0 or x_max <= k_last:
        return np.array([]), np.array([])
    k_ext = np.linspace(k_last, x_max, n)
    t_ext = t_last * (k_ext / k_last) ** power
    return k_ext, t_ext


def build_legend_elements(theory_power_label):
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=2.5,
            marker="o",
            label=legend_label(name),
        )
        for _, color, name in architectures
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=c_theory,
            ls="--",
            lw=1.5,
            label=theory_power_label,
        )
    )
    return handles


def compute_axis_limits(
    architectures_to_plot,
    time_col,
    y_pad_frac=0.06,
    top_reserve_frac=TOP_RESERVE_FRAC,
    extend_x_frac=EXTEND_X_FRAC,
):
    dfs = [df for df, _, _ in architectures_to_plot]
    ks, ts = collect_finite_points(dfs, time_col)
    if ks.size == 0:
        return 0.0, 300.0, 0.0, 1.0

    x_max = ks.max() * (1.0 + extend_x_frac)
    y_max = ts.max() * (1.0 + y_pad_frac) / max(1.0 - top_reserve_frac, 0.5)
    return 0.0, x_max, 0.0, y_max


def plot_formulation(
    time_col,
    power,
    theory_power_label,
    output_file,
):
    plotted = [(df, color, name) for df, color, name in architectures if not df.empty]
    x_min, x_max, y_min, y_max = compute_axis_limits(plotted, time_col)

    fig, ax = plt.subplots(figsize=fig_size, dpi=300)

    for df, color, _name in plotted:
        k_valid, t_valid = finite_kt(df, time_col)
        if k_valid.size == 0:
            continue

        last = last_finite_point(df, time_col)
        if last is not None:
            k_ext, t_ext = asymptotic_extension(last[0], last[1], power, x_max)
            if k_ext.size > 0:
                ax.plot(
                    k_ext,
                    t_ext,
                    color=c_theory,
                    ls="--",
                    lw=1.5,
                    alpha=0.9,
                    zorder=1,
                    clip_on=True,
                )

        ax.plot(
            k_valid,
            t_valid,
            color=color,
            ls="-",
            marker="o",
            markersize=8,
            lw=2.5,
            alpha=1.0,
            zorder=3,
            markevery=1,
        )

    set_mixed_xlabel(ax)
    ax.set_ylabel("Time per Iteration (s)", usetex=False)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    legend = ax.legend(
        handles=build_legend_elements(theory_power_label),
        loc="upper left",
        frameon=True,
        shadow=True,
    )
    if USE_TEX:
        legend.get_texts()[-1].set_usetex(True)

    plt.tight_layout()
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots():
    plot_formulation(
        time_col="time_N_IP",
        power=3,
        theory_power_label=cm_bold_big_o(3),
        output_file=str(OUT_DIR / "naive_performance.pdf"),
    )
    plot_formulation(
        time_col="time_S_IP",
        power=2,
        theory_power_label=cm_bold_big_o(2),
        output_file=str(OUT_DIR / "schur_performance.pdf"),
    )


try:
    make_plots()
except RuntimeError as err:
    # An incomplete LaTeX installation is found on PATH but cannot typeset the labels.
    if not USE_TEX:
        raise
    print("LaTeX could not render the labels; falling back to Matplotlib's mathtext.")
    print(f"  ({str(err).splitlines()[0]})")
    USE_TEX = False
    plt.close("all")
    make_plots()

print(f"Generated naive_performance.pdf and schur_performance.pdf in {OUT_DIR}")
