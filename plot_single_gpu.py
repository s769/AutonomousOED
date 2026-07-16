import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import seaborn as sns

# ==========================================
# 1. FILE CONFIGURATION
# ==========================================
DATA_DIR = Path(__file__).resolve().parent / "scaling_results"
FILE_A100 = DATA_DIR / "single_gpu_results_a100.csv"
FILE_MI250X = DATA_DIR / "single_gpu_results_mi250x.csv"
FILE_GH200 = DATA_DIR / "single_gpu_results_gh200.csv"
FILE_GB200 = DATA_DIR / "single_gpu_results_gb200.csv"
FILE_MI300X = DATA_DIR / "single_gpu_results_mi300x.csv"

FIT_K_MIN = 50
THEORY_ANCHOR_K = 100
TOP_RESERVE_FRAC = 0.28

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

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

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
        mask = np.isfinite(t) & (t > 0.0)
        ks.extend(k[mask])
        ts.extend(t[mask])
    return np.asarray(ks, dtype=float), np.asarray(ts, dtype=float)


def fit_loglog_exponent(k_array, time_array, k_min=FIT_K_MIN, k_max=None):
    k = np.asarray(k_array, dtype=float)
    t = np.asarray(time_array, dtype=float)
    mask = np.isfinite(k) & np.isfinite(t) & (t > 0.0) & (k >= k_min)
    if k_max is not None:
        mask &= k <= k_max
    if mask.sum() < 3:
        return np.nan
    return np.polyfit(np.log(k[mask]), np.log(t[mask]), 1)[0]


def legend_label(name, exponent):
    short = LEGEND_SHORT_NAMES.get(name, name)
    if np.isfinite(exponent):
        return rf"{short} ($k^{{{exponent:.2f}}}$)"
    return short


def build_legend_elements(time_col, theory_power_label):
    handles = []
    for df, color, name in architectures:
        exponent = np.nan
        if not df.empty:
            exponent = fit_loglog_exponent(df["k"], df[time_col])
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2.5,
                marker="o",
                label=legend_label(name, exponent),
            )
        )
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
    dfs,
    time_col,
    x_pad_frac=0.03,
    y_pad_frac=0.10,
    top_reserve_frac=TOP_RESERVE_FRAC,
):
    ks, ts = collect_finite_points(dfs, time_col)
    if ks.size == 0:
        return 0.0, 300.0, 0.0, 1.0

    x_max = ks.max() * (1.0 + x_pad_frac)
    y_data_max = ts.max() * (1.0 + y_pad_frac)
    y_max = y_data_max / max(1.0 - top_reserve_frac, 0.5)
    return 0.0, x_max, 0.0, y_max


def get_shared_theoretical_curve(
    dfs, time_col, power, x_min, x_max, anchor_k=THEORY_ANCHOR_K
):
    anchor_times = []
    for df in dfs:
        if df.empty:
            continue
        k = df["k"].to_numpy(dtype=float)
        t = df[time_col].to_numpy(dtype=float)
        mask = np.isfinite(k) & np.isfinite(t) & (t > 0.0)
        if not mask.any():
            continue
        idx = np.argmin(np.abs(k[mask] - anchor_k))
        anchor_times.append(t[mask][idx])

    if not anchor_times:
        return np.array([]), np.array([])

    t_anchor = float(np.median(anchor_times))
    c = t_anchor / (anchor_k**power)
    k_smooth = np.linspace(x_min, x_max, 100)
    return k_smooth, c * (k_smooth**power)


def plot_formulation(
    time_col,
    power,
    theory_power_label,
    output_file,
    x_pad_frac=0.03,
    y_pad_frac=0.10,
):
    dfs = [df for df, _, _ in architectures if not df.empty]
    x_min, x_max, y_min, y_max = compute_axis_limits(
        dfs, time_col, x_pad_frac=x_pad_frac, y_pad_frac=y_pad_frac
    )

    fig, ax = plt.subplots(figsize=fig_size, dpi=300)

    k_smooth, t_ideal = get_shared_theoretical_curve(
        dfs, time_col, power, x_min, x_max
    )
    if k_smooth.size > 0:
        ax.plot(
            k_smooth,
            t_ideal,
            color=c_theory,
            ls="--",
            lw=1.5,
            alpha=0.9,
            zorder=1,
        )

    for df, color, _name in architectures:
        if df.empty:
            continue

        ax.plot(
            df["k"],
            df[time_col],
            color=color,
            ls="-",
            marker="o",
            markersize=8,
            lw=2.5,
            alpha=1.0,
            zorder=3,
            markevery=1,
        )

    ax.set_xlabel(r"Selected Sensors ($k$)")
    ax.set_ylabel("Time per Iteration (s)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.legend(
        handles=build_legend_elements(time_col, theory_power_label),
        loc="upper left",
        frameon=True,
        shadow=True,
    )

    plt.tight_layout()
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close(fig)


plot_formulation(
    time_col="time_N_IP",
    power=3,
    theory_power_label=r"O($k^3$)",
    output_file="naive_performance.pdf",
)

plot_formulation(
    time_col="time_S_IP",
    power=2,
    theory_power_label=r"O($k^2$)",
    output_file="schur_performance.pdf",
)

print("Generated naive_performance.pdf and schur_performance.pdf")
