import h5py
import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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


def fetch_block(h5_dset, Nt, row_sensor, col_sensor, device, dtype):
    """Return the Nt x Nt block K[row_sensor, col_sensor] on the compute device.

    Only the upper triangle of K is read from disk; lower-triangle blocks are
    the transposes of their mirror blocks.
    """
    fetch_row, fetch_col = min(row_sensor, col_sensor), max(row_sensor, col_sensor)
    block = torch.from_numpy(get_numpy_block(h5_dset, Nt, fetch_row, fetch_col)).to(
        device=device, dtype=dtype
    )
    if row_sensor > col_sensor:
        block = block.T
    return block


def solve_lower_blocked(L, K_Sc, k, Nt):
    """Solve L[:k*Nt, :k*Nt] Y = K_Sc by block forward substitution, in place in K_Sc.

    Each step uses one Nt x Nt triangular solve and one matrix product, so no
    copy of the large factor is ever made.
    """
    for i in range(k):
        r0, r1 = i * Nt, (i + 1) * Nt
        if i > 0:
            K_Sc[r0:r1] -= L[r0:r1, :r0] @ K_Sc[:r0]
        K_Sc[r0:r1] = torch.linalg.solve_triangular(
            L[r0:r1, r0:r1], K_Sc[r0:r1], upper=False
        )
    return K_Sc

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
    """Evaluate the D-optimal objective log det(I + r^2 K_S) of a configuration.

    The Cholesky factor of I + r^2 K_S is built one sensor block at a time with
    Schur-complement updates, as oed_hist.py does for the random configurations.
    The dense matrix is never assembled or factorized in a single LAPACK call:
    doing that (the previous slogdet) needs a second full-size copy on the GPU
    and failed with a memory access fault on some ROCm installations.
    Peak GPU memory is one (B*Nt) x (B*Nt) factor plus a few (B*Nt) x Nt blocks.
    """
    n = len(config)
    size = n * Nt
    print(f"Factorizing {size}x{size} covariance matrix block by block on {device}...")
    L = torch.zeros((size, size), dtype=dtype, device=device)
    log_det = 0.0

    for k, c in enumerate(tqdm(config, desc="Sensors")):
        r0, r1 = k * Nt, (k + 1) * Nt

        K_cc = fetch_block(h5_dset, Nt, c, c, device, dtype)
        K_cc = 0.5 * (K_cc + K_cc.T)  # symmetrize to remove upstream float noise
        K_cc.mul_(r_sq).diagonal().add_(1.0)

        if k > 0:
            K_Sc = torch.empty((r0, Nt), dtype=dtype, device=device)
            for i, s in enumerate(config[:k]):
                K_Sc[i * Nt : (i + 1) * Nt] = fetch_block(h5_dset, Nt, s, c, device, dtype)
            K_Sc.mul_(r_sq)

            Y = solve_lower_blocked(L, K_Sc, k, Nt)
            K_cc -= Y.T @ Y
            L[r0:r1, :r0] = Y.T
            del K_Sc, Y

        try:
            L_cc = torch.linalg.cholesky(K_cc)
        except torch.linalg.LinAlgError:
            print(f"Sensor {c} (position {k}) is numerically dependent on the previous ones.")
            return -np.inf

        L[r0:r1, r0:r1] = L_cc
        log_det += 2.0 * torch.sum(torch.log(torch.diagonal(L_cc))).item()

    return log_det


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
    parser.add_argument("--coords_file", type=str, default="600_sensor_coords.csv", help="CSV for plotting")
    parser.add_argument("--budget", type=int, default=175, help="Strict target budget (B)")
    parser.add_argument("--n_lat", type=int, default=50, help="Number of contiguous latitudes")
    parser.add_argument("--n_lon", type=int, default=12, help="Number of longitudes")
    parser.add_argument("--r_sq", type=float, default=1.0, help="Scaling factor r^2")
    parser.add_argument("--precision", type=str, choices=["single", "double"], default="single")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic add/drop")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device (auto: first GPU if one is visible, otherwise CPU)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="uniform_result.txt",
        help="Text file that receives the objective value (read by oed_hist.py --uniform_file)",
    )
    
    args = parser.parse_args()

    # Hardware Setup
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Error: --device cuda was requested but PyTorch cannot see a GPU.")
        return
    if args.device != "cpu" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    torch.set_grad_enabled(False)

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

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(f"{score:.18e}\n")
        print(f"Objective value written to '{args.output_file}'")

    h5_file.close()

if __name__ == "__main__":
    main()