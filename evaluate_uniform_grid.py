import h5py
import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Standardized Styling ---
sns.set_context("paper", font_scale=1.3)
sns.set_style("whitegrid")
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


def get_numpy_block(h5_dset, Nt, block_row_idx, block_col_idx):
    """Fetches a specific block from the chunked HDF5 matrix."""
    r_start, c_start = int(block_row_idx * Nt), int(block_col_idx * Nt)
    block = h5_dset[r_start : r_start + Nt, c_start : c_start + Nt]
    if np.isnan(block).any():
        block = np.nan_to_num(block, nan=0.0)
    return block


def build_k_submatrix(h5_dset, Nt, S_indices, device, dtype, r_sq=1.0):
    """Builds the dense K_S submatrix for the selected sensors."""
    current_size = len(S_indices) * Nt
    K_S = torch.zeros((current_size, current_size), dtype=dtype, device=device)

    print(f"Building {current_size}x{current_size} covariance matrix on {device}...")
    
    for row_idx, s_row in enumerate(S_indices):
        for col_idx in range(row_idx, len(S_indices)):
            s_col = S_indices[col_idx]

            fetch_row, fetch_col = min(s_row, s_col), max(s_row, s_col)
            block_np = get_numpy_block(h5_dset, Nt, fetch_row, fetch_col)
            block = torch.from_numpy(block_np).to(device=device, dtype=dtype)

            # Symmetrize diagonal blocks to prevent upstream float noise
            if fetch_row == fetch_col:
                block = 0.5 * (block + block.T)

            if s_row > s_col:
                block = block.T

            block.mul_(r_sq)
            r_start, c_start = row_idx * Nt, col_idx * Nt

            K_S[r_start : r_start + Nt, c_start : c_start + Nt] = block
            if col_idx > row_idx:
                K_S[c_start : c_start + Nt, r_start : r_start + Nt] = block.T

    K_S.diagonal().add_(1.0)
    return K_S


def uniform_subsample_indices(n_total, n_select):
    """Pick n_select evenly spaced indices from [0, n_total - 1].

    When n_total divides n_select evenly, this uses an exact stride (e.g. 6 of 12
    -> every other column: 0, 2, 4, 6, 8, 10). Otherwise falls back to rounded
    linspace over the endpoints.
    """
    if n_select <= 0:
        return np.array([], dtype=int)
    if n_select >= n_total:
        return np.arange(n_total, dtype=int)
    if n_select == 1:
        return np.array([(n_total - 1) // 2], dtype=int)
    if n_total % n_select == 0:
        stride = n_total // n_select
        return np.arange(0, n_total, stride, dtype=int)[:n_select]
    return np.round(np.linspace(0, n_total - 1, n_select)).astype(int)


def pick_center_fill_sensors(unselected, add_count, n_lon, n_lat):
    """Pick unselected sensors closest to the grid interior (away from edges)."""
    lon_c = (n_lon - 1) / 2.0
    lat_c = (n_lat - 1) / 2.0

    def placement_score(idx):
        lon, lat = idx // n_lat, idx % n_lat
        edge_penalty = 0
        if lon == 0 or lon == n_lon - 1:
            edge_penalty += 100
        if lat == 0 or lat == n_lat - 1:
            edge_penalty += 100
        dist = (lon - lon_c) ** 2 + (lat - lat_c) ** 2
        return (edge_penalty, dist)

    ranked = sorted(unselected, key=placement_score)
    return ranked[:add_count]


def evaluate_config(config, h5_dset, Nt, device, dtype, r_sq=1.0):
    """Evaluate the D-optimal objective (log-det) of a configuration."""
    K_S = build_k_submatrix(h5_dset, Nt, config, device, dtype, r_sq)

    sign, logabsdet = torch.linalg.slogdet(K_S)
    if sign <= 0:
        return -np.inf
    return logabsdet.item()


def plot_grid_selection(coords_file, selected_indices, actual_budget):
    """Creates a spatial scatter plot of the full grid vs. the exact subset."""
    if not os.path.exists(coords_file):
        print(f"Coordinate file '{coords_file}' not found. Skipping plot.")
        return

    data = np.loadtxt(coords_file, delimiter=',', skiprows=1)
    lons = data[:, 0]
    lats = data[:, 1]
    
    selected_lons = lons[selected_indices]
    selected_lats = lats[selected_indices]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor('#0F0E0D')
    ax.set_facecolor('#0F0E0D')

    # Plot all 600 sensors as faint background dots
    ax.scatter(lons, lats, c='gray', alpha=0.3, s=10, label="Full Grid (600)")
    
    # Plot selected exact subset
    ax.scatter(selected_lons, selected_lats, c='#329874', alpha=1.0, s=40, 
               edgecolor='white', linewidth=0.5, label=f"Exact Uniform Subset ({actual_budget})")

    ax.set_xlabel("Longitude", color='white', fontweight='bold')
    ax.set_ylabel("Latitude", color='white', fontweight='bold')
    ax.set_title(f"Uniform Grid Approximation: Exactly {actual_budget} Sensors", color='white', fontweight='bold')
    
    ax.tick_params(colors='white')
    for spine in ['bottom', 'left']: ax.spines[spine].set_color('white')
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)

    legend = ax.legend(loc="best", frameon=True, facecolor='#0F0E0D', edgecolor='white')
    for text in legend.get_texts(): text.set_color("white")
    ax.grid(True, alpha=0.15, color='white')

    plt.tight_layout()
    plt.savefig("exact_uniform_grid.pdf", facecolor=fig.get_facecolor())
    print("Spatial map saved to 'exact_uniform_grid.pdf'")


def main():
    parser = argparse.ArgumentParser(description="Evaluate an Exact-Budget Uniform Grid Subsample")
    parser.add_argument("--h5_store_K", type=str, required=True, help="Path to HDF5 K Matrix")
    parser.add_argument("--coords_file", type=str, default="sensor_coords.csv", help="CSV for plotting")
    parser.add_argument("--budget", type=int, default=175, help="Strict target budget (B)")
    parser.add_argument("--n_lat", type=int, default=50, help="Number of contiguous latitudes")
    parser.add_argument("--n_lon", type=int, default=12, help="Number of longitudes")
    parser.add_argument("--r_sq", type=float, default=1.0, help="Scaling factor r^2")
    parser.add_argument("--precision", type=str, choices=["single", "double"], default="single")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic add/drop")
    
    args = parser.parse_args()

    # Hardware Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    compute_dtype = torch.float64 if args.precision == "double" else torch.float32

    # Verify grid dimensions
    total_grid_sensors = args.n_lat * args.n_lon
    print(f"Expected Full Grid: {args.n_lon} Lon x {args.n_lat} Lat = {total_grid_sensors} sensors")

    if args.budget > total_grid_sensors:
        print(f"Error: Budget ({args.budget}) exceeds total available sensors ({total_grid_sensors}).")
        return

    # Open HDF5
    try:
        h5_file = h5py.File(args.h5_store_K, mode="r", rdcc_nbytes=0)
        dset_name = "K_matrix" if "K_matrix" in h5_file else list(h5_file.keys())[0]
        h5_dset = h5_file[dset_name]
        Nt = h5_dset.chunks[0]
        Nd = h5_dset.shape[0] // Nt
        
        if Nd != total_grid_sensors:
            print(f"Warning: HDF5 has {Nd} sensors, but grid parameters expect {total_grid_sensors}.")
    except Exception as e:
        print(f"Error opening HDF5: {e}")
        return

    # -------------------------------------------------------------
    # 1. CALCULATE BEST 2D BASE GRID
    # -------------------------------------------------------------
    aspect_ratio = args.n_lon / args.n_lat
    b_lon = max(1, int(round(np.sqrt(args.budget * aspect_ratio))))
    b_lat = max(1, min(args.n_lat, args.budget // b_lon))
    base_budget = b_lon * b_lat

    print(f"\nTarget Budget: {args.budget}")
    print(f"Optimal 2D Base Grid: {b_lon} (lon) x {b_lat} (lat) = {base_budget} sensors.")

    idx_lon = uniform_subsample_indices(args.n_lon, b_lon)
    idx_lat = uniform_subsample_indices(args.n_lat, b_lat)
    print(f"Longitude column indices ({len(idx_lon)}): {idx_lon.tolist()}")
    print(f"Latitude row indices ({len(idx_lat)}): {idx_lat.tolist()}")

    mesh_lon, mesh_lat = np.meshgrid(idx_lon, idx_lat, indexing='ij')

    # Longitude is the outer (slow) index; latitude is contiguous within each column.
    selected_indices = (mesh_lon * args.n_lat + mesh_lat).flatten().tolist()

    # -------------------------------------------------------------
    # 2. ENFORCE EXACT BUDGET (Add or Drop)
    # -------------------------------------------------------------
    if len(selected_indices) != args.budget:
        print(f"Enforcing strict budget of {args.budget}...")
        rng = np.random.RandomState(args.seed) 
        
        if len(selected_indices) > args.budget:
            drop_count = len(selected_indices) - args.budget
            print(f" -> Uniform base is too large. Randomly dropping {drop_count} sensors.")
            selected_indices = rng.choice(selected_indices, args.budget, replace=False).tolist()
        else:
            add_count = args.budget - len(selected_indices)
            print(f" -> Uniform base is too small. Filling {add_count} gap(s) near grid center.")
            unselected = list(set(range(total_grid_sensors)) - set(selected_indices))
            extras = pick_center_fill_sensors(
                unselected, add_count, args.n_lon, args.n_lat
            )
            print(f" -> Added sensor index(es): {extras}")
            selected_indices.extend(extras)
            
    selected_indices.sort()

    # -------------------------------------------------------------
    # 3. EVALUATE & PLOT
    # -------------------------------------------------------------
    plot_grid_selection(args.coords_file, selected_indices, args.budget)

    print("\nStarting evaluation...")
    score = evaluate_config(selected_indices, h5_dset, Nt, device, compute_dtype, args.r_sq)
    
    print("\n" + "="*40)
    print("         UNIFORM GRID RESULT")
    print("="*40)
    print(f"Final Network Size    : {len(selected_indices)} sensors")
    print(f"D-Optimal Objective   : {score:.6e}")
    print("="*40 + "\n")

    h5_file.close()

if __name__ == "__main__":
    main()