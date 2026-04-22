import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import seaborn as sns

# ==========================================
# 1. FILE CONFIGURATION
# ==========================================
FILE_A100 = "test_new_a100.csv"
FILE_MI250X = "test_new_mi250x.csv"
FILE_GH200 = "test_new_gh200.csv"


def load_data(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        print(f"Warning: {filename} not found.")
        return pd.DataFrame(
            columns=["k", "time_N_OOP", "time_N_IP", "time_S_OOP", "time_S_IP"]
        )


df_a100 = load_data(FILE_A100)
df_mi250 = load_data(FILE_MI250X)
df_gh200 = load_data(FILE_GH200)

# ==========================================
# 2. PLOT STYLING & SETUP
# ==========================================
sns.set_context("paper", font_scale=1.3)
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("seaborn-whitegrid")

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

# Use native seaborn colorblind palette
cb = sns.color_palette("colorblind")
c_a100 = cb[0]  # Blue
c_mi250 = cb[1]  # Orange
c_gh200 = cb[2]  # Green
c_theory = "#666666"  # Universal Gray for all theoretical lines

# Find the global minimum and maximum k to extend the theoretical curves to the absolute edge
all_k = pd.concat([df_a100["k"], df_mi250["k"], df_gh200["k"]])
global_min_k = all_k.min() if not all_k.empty else 10
global_max_k = all_k.max() if not all_k.empty else 300
global_max_k += 20


def get_theoretical_curve(k_array, time_array, power):
    """Fits y = C * k^power and extends it to the global max k."""
    mask = np.isfinite(time_array)
    if not np.any(mask):
        return np.array([]), np.array([])

    valid_k = k_array[mask].values
    valid_t = time_array[mask].values

    # Use the max valid k to anchor the curve
    k_anchor = valid_k[-1]
    t_anchor = valid_t[-1]

    C = t_anchor / (k_anchor**power)

    # Extend smoothly to the absolute edge of the plot's X-axis
    k_smooth = np.linspace(global_min_k, global_max_k, 100)
    t_ideal = C * (k_smooth**power)
    return k_smooth, t_ideal


fig_size = (5.0, 4.0)

architectures = [
    (df_a100, c_a100, "NVIDIA A100"),
    (df_mi250, c_mi250, "AMD MI250X"),
    (df_gh200, c_gh200, "NVIDIA GH200"),
]

# Standard legend elements for both plots
legend_elements = [
    Line2D([0], [0], color=c_a100, lw=2.5, marker="o", label="NVIDIA A100"),
    Line2D([0], [0], color=c_mi250, lw=2.5, marker="o", label="AMD MI250X"),
    Line2D([0], [0], color=c_gh200, lw=2.5, marker="o", label="NVIDIA GH200"),
]

# ==========================================
# 3. PLOT 1: NAIVE FORMULATION (O(k^3))
# ==========================================
fig1, ax1 = plt.subplots(figsize=fig_size, dpi=300)

for df, color, name in architectures:
    if not df.empty:
        # Theoretical O(k^3) Fit
        k_smooth, t_ideal = get_theoretical_curve(df["k"], df["time_N_IP"], power=3)
        if k_smooth.size > 0:
            ax1.plot(
                k_smooth, t_ideal, color=c_theory, ls="--", lw=1.5, alpha=0.8, zorder=1
            )

        # Naive: Solid Line, Filled Circles
        ax1.plot(
            df["k"],
            df["time_N_IP"],
            color=color,
            ls="-",
            marker="o",
            markersize=8,
            lw=2.5,
            alpha=1.0,
            zorder=3,
            markevery=1,
        )

ax1.set_xlabel(r"Selected Sensors ($k$)")
ax1.set_ylabel("Time per Iteration (s)")

legend_naive = legend_elements + [
    Line2D([0], [0], color=c_theory, ls="--", lw=1.5, label=r"O($k^3$)")
]
ax1.legend(handles=legend_naive, loc="upper left", frameon=True, shadow=True)
ax1.set_xlim(0, global_max_k)

plt.tight_layout()
fig1.savefig("naive_performance.pdf", format="pdf", bbox_inches="tight")


# ==========================================
# 4. PLOT 2: SCHUR FORMULATION (O(k^2))
# ==========================================
fig2, ax2 = plt.subplots(figsize=fig_size, dpi=300)

for df, color, name in architectures:
    if not df.empty:
        # Theoretical O(k^2) Fit
        k_smooth, t_ideal = get_theoretical_curve(df["k"], df["time_S_IP"], power=2)
        if k_smooth.size > 0:
            ax2.plot(
                k_smooth, t_ideal, color=c_theory, ls="--", lw=1.5, alpha=0.8, zorder=1
            )

        # Schur: Solid Line, Filled Circles
        ax2.plot(
            df["k"],
            df["time_S_IP"],
            color=color,
            ls="-",
            marker="o",
            markersize=8,
            lw=2.5,
            alpha=1.0,
            zorder=3,
            markevery=1,
        )

ax2.set_xlabel(r"Selected Sensors ($k$)")
ax2.set_ylabel("Time per Iteration (s)")

legend_schur = legend_elements + [
    Line2D([0], [0], color=c_theory, ls="--", lw=1.5, label=r"O($k^2$)")
]
ax2.legend(handles=legend_schur, loc="upper left", frameon=True, shadow=True)
ax2.set_xlim(0, global_max_k)

plt.tight_layout()
fig2.savefig("schur_performance.pdf", format="pdf", bbox_inches="tight")

print("Generated naive_performance.pdf and schur_performance.pdf")
