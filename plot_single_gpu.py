import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import seaborn as sns
import os
import shutil

# ==========================================
# 1. FILE CONFIGURATION
# ==========================================
DATA_DIR = Path(__file__).resolve().parent / "scaling_results"
FILE_A100 = DATA_DIR / "single_gpu_results_a100.csv"
FILE_MI250X = DATA_DIR / "single_gpu_results_mi250x.csv"
FILE_GH200 = DATA_DIR / "single_gpu_results_gh200.csv"
FILE_GB200 = DATA_DIR / "single_gpu_results_gb200.csv"
FILE_MI300X = DATA_DIR / "single_gpu_results_mi300x.csv"

TOP_RESERVE_FRAC = 0.08
EXTEND_X_FRAC = 0.20

LEGEND_SHORT_NAMES = {
    "AMD MI250X": "MI250X",
    "NVIDIA A100": "A100",
    "NVIDIA GH200": "GH200",
    "NVIDIA GB200": "GB200",
    "NVIDIA MI300X": "MI300X",
}

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
USE_TEX = shutil.which("latex") is not None

plt.rcParams["text.usetex"] = False
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.default"] = "it"
# Embed TrueType (Type 42) instead of Type 3 bitmap fonts for publisher PDFs.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
if USE_TEX:
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{bm}"

cb = sns.color_palette("colorblind")
c_mi250 = cb[1]
c_a100 = cb[0]
c_gh200 = cb[2]
c_gb200 = cb[3]
c_mi300x = cb[4]
c_theory = "#666666"

fig_size = (5.0, 4.0)


def load_data(filename):
    if filename.exists():
        return pd.read_csv(filename)
    print(f"Warning: {filename} not found.")
    return pd.DataFrame(
        columns=["k", "time_N_OOP", "time_N_IP", "time_S_OOP", "time_S_IP"]
    )


df_mi250 = load_data(FILE_MI250X)
df_a100 = load_data(FILE_A100)
df_gh200 = load_data(FILE_GH200)
df_gb200 = load_data(FILE_GB200)
df_mi300x = load_data(FILE_MI300X)
architectures = [
    (df_mi250, c_mi250, "AMD MI250X"),
    (df_a100, c_a100, "NVIDIA A100"),
    (df_gh200, c_gh200, "NVIDIA GH200"),
    (df_gb200, c_gb200, "NVIDIA GB200"),
    (df_mi300x, c_mi300x, "NVIDIA MI300X"),
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
    return LEGEND_SHORT_NAMES.get(name, name)


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


plot_formulation(
    time_col="time_N_IP",
    power=3,
    theory_power_label=cm_bold_big_o(3),
    output_file=str(DATA_DIR / "naive_performance.pdf"),
)

plot_formulation(
    time_col="time_S_IP",
    power=2,
    theory_power_label=cm_bold_big_o(2),
    output_file=str(DATA_DIR / "schur_performance.pdf"),
)

print("Generated naive_performance.pdf and schur_performance.pdf")
