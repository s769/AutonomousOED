import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import seaborn as sns

sns.set_context("talk")
sns.set_style("whitegrid")
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot and animate standard D-optimal sensor selection."
    )
    parser.add_argument(
        "--sensors",
        default="sens-mw-300-12x50-150-latlon.csv",
        help="CSV of candidate sensor coordinates (columns: lon, lat).",
    )
    parser.add_argument(
        "--indices",
        default="oed-200.txt",
        help="Whitespace-separated file of selected sensor indices and objective values.",
    )
    return parser.parse_args()


def plot_sensor_map(ax, df_all, df_selected, title_text):
    ax.scatter(df_all['lon'], df_all['lat'], c='lightgray', s=20, label='All Sensors')
    ax.scatter(df_selected['lon'], df_selected['lat'], c='red', s=40, zorder=3, label='Selected Sensors')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title_text)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.6)


def main():
    args = parse_args()

    df_all = pd.read_csv(args.sensors)
    df_oed_info = pd.read_csv(
        args.indices, sep=r'\s+', header=None, names=['index', 'objective'], engine='python',
    )

    oed_valid_indices = [idx for idx in df_oed_info['index'] if idx < len(df_all)]
    df_selected_oed = df_all.loc[oed_valid_indices].copy()
    n_selected = len(df_oed_info)

    # --- FIGURE 1: Sensor locations ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plot_sensor_map(ax1, df_all, df_selected_oed, 'D-Optimal Sensor Selection')
    plt.savefig("sensor_locations_standard.pdf")

    # --- FIGURE 2: Objective function ---
    fig2, ax_left = plt.subplots(figsize=(12, 8))
    num_sensors = np.arange(1, n_selected + 1)
    color_oed = 'teal'
    ax_left.set_xlabel('Number of Sensors')
    ax_left.set_ylabel('Standard Objective', color=color_oed)
    ax_left.plot(
        num_sensors, df_oed_info['objective'],
        marker='o', linestyle='-', color=color_oed, label='Standard Objective', markevery=5,
    )
    ax_left.tick_params(axis='y', labelcolor=color_oed)
    ax_left.grid(True, linestyle='--', alpha=0.6)
    ax_left.legend(loc='best', frameon=True, shadow=True)
    ax_left.set_title('Standard Objective Function')

    num_ticks = min(10, len(num_sensors))
    if num_ticks > 0:
        tick_locs = np.linspace(1, n_selected, num_ticks, dtype=int)
        if 1 not in tick_locs and n_selected > 1:
            tick_locs[0] = 1
        ax_left.set_xticks(tick_locs)

    fig2.tight_layout()
    plt.savefig("objective_functions_standard.pdf")

    # --- FIGURE 3: Animation of sensor selection ---
    print("Generating animation... this may take a moment.")
    animation_duration_seconds = 15
    interval_ms = (animation_duration_seconds * 1000) / n_selected

    fig_anim, anim_ax1 = plt.subplots(figsize=(8, 6))
    anim_ax1.scatter(df_all['lon'], df_all['lat'], c='lightgray', s=20, label='All Sensors')
    anim_ax1.set_xlabel('Longitude')
    anim_ax1.set_ylabel('Latitude')
    anim_ax1.set_title('Standard D-Optimal Selection')
    anim_ax1.grid(True, linestyle='--', alpha=0.6)
    scat_oed = anim_ax1.scatter([], [], c='red', s=40, zorder=3, label='Selected Sensors')
    text_oed = anim_ax1.text(0.95, 0.95, '', transform=anim_ax1.transAxes, ha='right', va='top')
    anim_ax1.legend(loc='best', frameon=True, shadow=True)

    def init():
        scat_oed.set_offsets(np.empty((0, 2)))
        text_oed.set_text('')
        return scat_oed, text_oed

    def update(frame):
        current_oed = df_selected_oed.iloc[:frame + 1]
        scat_oed.set_offsets(current_oed[['lon', 'lat']])
        text_oed.set_text(f'Sensors: {frame + 1}/{n_selected}')
        return scat_oed, text_oed

    ani = animation.FuncAnimation(
        fig_anim, update, frames=n_selected,
        init_func=init, blit=True, interval=interval_ms,
    )

    ani.save('sensor_selection_standard.gif', writer='pillow')
    ani.save('sensor_selection_standard.mp4', writer='ffmpeg', fps=15)

    print("Animation saved to sensor_selection_standard.gif and .mp4")
    plt.show()
    print("Plots and animation have been saved.")


if __name__ == '__main__':
    main()
