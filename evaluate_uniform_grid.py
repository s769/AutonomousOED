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
# Embed TrueType (Type 42) instead of Type 3 bitmap fonts for publisher PDFs.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


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


def load_sensor_lonlat(coords_file):
    """Load (Nd, 2) lon/lat array from CSV (header + lon,lat,...)."""
    if not os.path.exists(coords_file):
        raise FileNotFoundError(f"Coordinate file '{coords_file}' not found.")
    data = np.loadtxt(coords_file, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected lon/lat columns in {coords_file}")
    return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)


def haversine_km_matrix(lons, lats):
    """Pairwise great-circle distances (km) for lon/lat in degrees."""
    lon = np.radians(lons)
    lat = np.radians(lats)
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def maximin_farthest_point(lons, lats, budget, seed=42):
    """Exact-budget space-filling design via farthest-point (Gonzalez) sampling.

    Starts at the sensor farthest from the geographic centroid, then repeatedly
    adds the unused site that maximizes distance to the current set (haversine km).
    Ties broken by smallest index for determinism (seed kept for API compatibility).
    """
    del seed  # deterministic without RNG; retained for CLI compatibility
    Nd = len(lons)
    if budget <= 0:
        return []
    if budget >= Nd:
        return list(range(Nd))

    D = haversine_km_matrix(lons, lats)
    lon_c, lat_c = float(np.mean(lons)), float(np.mean(lats))
    # Approximate centroid distance with haversine to a synthetic point via matrix
    # of each site vs itself is 0; compute vs centroid directly:
    R = 6371.0
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    lat0 = np.radians(lat_c)
    lon0 = np.radians(lon_c)
    dlat = lat_r - lat0
    dlon = lon_r - lon0
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_r) * np.cos(lat0) * np.sin(dlon / 2.0) ** 2
    dist_centroid = R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    first = int(np.argmax(dist_centroid))

    selected = [first]
    min_dist = D[first].copy()
    min_dist[first] = -np.inf

    for _ in range(1, budget):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, D[nxt])
        min_dist[nxt] = -np.inf

    return selected


def rectangular_grid_selection(budget, n_lon, n_lat, seed=42):
    """Legacy aspect-matched product grid with exact-budget add/drop."""
    total_grid_sensors = n_lat * n_lon
    aspect_ratio = n_lon / n_lat
    b_lon = max(1, int(round(np.sqrt(budget * aspect_ratio))))
    b_lat = max(1, min(n_lat, budget // b_lon))
    base_budget = b_lon * b_lat

    print(f"Optimal 2D Base Grid: {b_lon} (lon) x {b_lat} (lat) = {base_budget} sensors.")

    idx_lon = uniform_subsample_indices(n_lon, b_lon)
    idx_lat = uniform_subsample_indices(n_lat, b_lat)
    print(f"Longitude column indices ({len(idx_lon)}): {idx_lon.tolist()}")
    print(f"Latitude row indices ({len(idx_lat)}): {idx_lat.tolist()}")

    mesh_lon, mesh_lat = np.meshgrid(idx_lon, idx_lat, indexing="ij")
    selected_indices = (mesh_lon * n_lat + mesh_lat).flatten().tolist()

    if len(selected_indices) != budget:
        print(f"Enforcing strict budget of {budget}...")
        rng = np.random.RandomState(seed)
        if len(selected_indices) > budget:
            drop_count = len(selected_indices) - budget
            print(f" -> Uniform base is too large. Randomly dropping {drop_count} sensors.")
            selected_indices = rng.choice(selected_indices, budget, replace=False).tolist()
        else:
            add_count = budget - len(selected_indices)
            print(f" -> Uniform base is too small. Filling {add_count} gap(s) near grid center.")
            unselected = list(set(range(total_grid_sensors)) - set(selected_indices))
            extras = pick_center_fill_sensors(unselected, add_count, n_lon, n_lat)
            print(f" -> Added sensor index(es): {extras}")
            selected_indices.extend(extras)
    return selected_indices


def evenly_spaced_indices(n_total, n_select):
    """Spread n_select indices across [0, n_total) with as-equal gaps as possible.

    Uses bin centers (not forced endpoints), which avoids the endpoint-anchored
    linspace pattern that packs some samples into adjacent slots.
    """
    if n_select <= 0:
        return np.array([], dtype=int)
    if n_select >= n_total:
        return np.arange(n_total, dtype=int)
    if n_select == 1:
        return np.array([n_total // 2], dtype=int)
    return np.array(
        [int((i + 0.5) * n_total / n_select) for i in range(n_select)],
        dtype=int,
    )


def _maximin_fill(selected, n_total, n_add, lons, lats):
    """Add n_add unused sensors by geographic farthest-point."""
    if n_add <= 0:
        return list(selected)
    selected = list(selected)
    remaining = [i for i in range(n_total) if i not in set(selected)]
    D = haversine_km_matrix(lons, lats)
    for _ in range(n_add):
        if not remaining:
            break
        sel = np.asarray(selected, dtype=int)
        rem = np.asarray(remaining, dtype=int)
        dmin = D[np.ix_(rem, sel)].min(axis=1) if len(sel) else np.zeros(len(rem))
        pick = int(rem[int(np.argmax(dmin))])
        selected.append(pick)
        remaining.remove(pick)
    return selected


def staggered_lattice_selection(budget, n_lon, n_lat, lons=None, lats=None):
    """Stride-2 hexagonal (brick) sublattice of the original sensor grid.

    Uses every-other longitude in alternating phases:
      even lat rows: 0,2,4,...   odd lat rows: 1,3,5,...
    so no two selected sensors are longitude-adjacent within a row.
    Latitude rows are bin-centered and evenly spaced. If the product is short
    of ``budget`` (e.g. 6*29=174 for 175), remaining slots are filled by
    geographic farthest-point among unused sites (usually one sensor).
    """
    if n_lon % 2 != 0:
        raise ValueError(
            f"Stride-2 hex stagger requires even n_lon (got {n_lon}). "
            "Use --method rectangular or maximin."
        )

    n_per_row = n_lon // 2
    n_lat_sel = max(1, min(n_lat, budget // n_per_row))
    base_budget = n_per_row * n_lat_sel
    idx_lat = evenly_spaced_indices(n_lat, n_lat_sel)
    lon_even = np.arange(0, n_lon, 2, dtype=int)
    lon_odd = np.arange(1, n_lon, 2, dtype=int)

    print(
        f"Hex stagger (stride-2): {n_per_row} lon/row x {n_lat_sel} lat = "
        f"{base_budget} lattice sensors."
    )
    print(f"Latitude row indices ({len(idx_lat)}): {idx_lat.tolist()}")
    print(f"Lon phase A (even rows): {lon_even.tolist()}")
    print(f"Lon phase B (odd rows):  {lon_odd.tolist()}")
    if len(idx_lat) > 1:
        lat_gaps = np.diff(idx_lat)
        print(
            f"Lat gap range: [{lat_gaps.min()}, {lat_gaps.max()}] "
            "(no zero-gaps; lon spacing is exactly 2)"
        )

    selected = []
    for row_i, lat in enumerate(idx_lat):
        lons_row = lon_even if (row_i % 2 == 0) else lon_odd
        for lon in lons_row:
            selected.append(int(lon) * n_lat + int(lat))

    if len(set(selected)) != len(selected):
        raise RuntimeError("Hex stagger produced duplicate sensor indices.")

    n_fill = budget - len(selected)
    if n_fill < 0:
        raise RuntimeError(f"Hex lattice oversized: {len(selected)} > budget {budget}")
    if n_fill > 0:
        if lons is None or lats is None:
            raise ValueError(
                f"Hex lattice has {len(selected)} sensors; need coords to fill "
                f"{n_fill} to reach budget {budget}."
            )
        print(f"Filling {n_fill} leftover slot(s) by geographic maximin...")
        before = set(selected)
        selected = _maximin_fill(selected, n_lon * n_lat, n_fill, lons, lats)
        extras = [i for i in selected if i not in before]
        print(f" -> Added sensor index(es): {extras}")

    if len(selected) != budget:
        raise RuntimeError(
            f"Hex stagger produced {len(selected)} sensors, expected {budget}."
        )
    return selected


def evaluate_config(config, h5_dset, Nt, device, dtype, r_sq=1.0):
    """Evaluate the D-optimal objective (log-det) of a configuration."""
    K_S = build_k_submatrix(h5_dset, Nt, config, device, dtype, r_sq)

    sign, logabsdet = torch.linalg.slogdet(K_S)
    if sign <= 0:
        return -np.inf
    return logabsdet.item()


def plot_grid_selection(coords_file, selected_indices, actual_budget, plot_path, title=None):
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
    ax.scatter(lons, lats, c='gray', alpha=0.3, s=10, label=f"Full Grid ({len(lons)})")
    
    # Plot selected exact subset
    ax.scatter(selected_lons, selected_lats, c='#329874', alpha=1.0, s=40, 
               edgecolor='white', linewidth=0.5, label=f"Selected ({actual_budget})")

    ax.set_xlabel("Longitude", color='white', fontweight='bold')
    ax.set_ylabel("Latitude", color='white', fontweight='bold')
    ax.set_title(
        title or f"Space-filling design: Exactly {actual_budget} Sensors",
        color='white',
        fontweight='bold',
    )
    
    ax.tick_params(colors='white')
    for spine in ['bottom', 'left']: ax.spines[spine].set_color('white')
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)

    legend = ax.legend(loc="best", frameon=True, facecolor='#0F0E0D', edgecolor='white')
    for text in legend.get_texts(): text.set_color("white")
    ax.grid(True, alpha=0.15, color='white')

    plt.tight_layout()
    plt.savefig(plot_path, facecolor=fig.get_facecolor())
    print(f"Spatial map saved to '{plot_path}'")


def main():
    parser = argparse.ArgumentParser(description="Evaluate an Exact-Budget Uniform Grid Subsample")
    parser.add_argument("--h5_store_K", type=str, required=True, help="Path to HDF5 K Matrix")
    parser.add_argument("--coords_file", type=str, default="sensor_coords.csv", help="CSV for plotting / maximin")
    parser.add_argument("--budget", type=int, default=175, help="Strict target budget (B)")
    parser.add_argument("--n_lat", type=int, default=50, help="Number of contiguous latitudes")
    parser.add_argument("--n_lon", type=int, default=12, help="Number of longitudes")
    parser.add_argument("--r_sq", type=float, default=1.0, help="Scaling factor r^2")
    parser.add_argument("--precision", type=str, choices=["single", "double"], default="single")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic add/drop")
    parser.add_argument(
        "--method",
        type=str,
        choices=["hex", "staggered", "maximin", "rectangular"],
        default="hex",
        help=(
            "Sensor placement (default: hex). "
            "'hex'/'staggered': stride-2 brick lattice on the original grid "
            "(even/odd lon phases, evenly spaced lats, maximin fill to exact budget); "
            "'maximin': farthest-point; "
            "'rectangular': legacy aspect-matched product grid."
        ),
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default="exact_uniform_grid.pdf",
        help="Output path for the spatial selection plot.",
    )
    parser.add_argument(
        "--selection_out",
        type=str,
        default=None,
        help="Optional path to write selected sensor indices (one per line).",
    )
    
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
    print(f"Method: {args.method}")

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

    print(f"\nTarget Budget: {args.budget}")
    if args.method == "maximin":
        lons, lats = load_sensor_lonlat(args.coords_file)
        if len(lons) != Nd:
            print(f"Warning: coords have {len(lons)} rows but HDF5 has Nd={Nd}.")
        selected_indices = maximin_farthest_point(lons, lats, args.budget, seed=args.seed)
        D = haversine_km_matrix(lons, lats)
        sel = np.asarray(selected_indices, dtype=int)
        sub = D[np.ix_(sel, sel)]
        np.fill_diagonal(sub, np.inf)
        print(f"Maximin selection: {len(selected_indices)} sensors")
        print(f"Min pairwise spacing among selected: {sub.min():.3f} km")
        plot_title = f"Maximin (farthest-point): Exactly {args.budget} Sensors"
    elif args.method in ("hex", "staggered"):
        lons, lats = load_sensor_lonlat(args.coords_file)
        selected_indices = staggered_lattice_selection(
            args.budget, args.n_lon, args.n_lat, lons=lons, lats=lats
        )
        D = haversine_km_matrix(lons, lats)
        sel = np.asarray(selected_indices, dtype=int)
        sub = D[np.ix_(sel, sel)]
        np.fill_diagonal(sub, np.inf)
        print(f"Min pairwise spacing among selected: {sub.min():.3f} km")
        plot_title = f"Hex stagger (stride-2): Exactly {args.budget} Sensors"
    else:
        selected_indices = rectangular_grid_selection(
            args.budget, args.n_lon, args.n_lat, seed=args.seed
        )
        plot_title = f"Rectangular lattice: Exactly {args.budget} Sensors"

    selected_sorted = sorted(selected_indices)

    if args.selection_out:
        out_dir = os.path.dirname(os.path.abspath(args.selection_out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savetxt(args.selection_out, np.asarray(selected_sorted, dtype=int), fmt="%d")
        print(f"Wrote selection to {args.selection_out}")

    # -------------------------------------------------------------
    # EVALUATE & PLOT
    # -------------------------------------------------------------
    plot_grid_selection(
        args.coords_file,
        selected_sorted,
        args.budget,
        plot_path=args.plot_path,
        title=plot_title,
    )

    print("\nStarting evaluation...")
    score = evaluate_config(selected_sorted, h5_dset, Nt, device, compute_dtype, args.r_sq)
    
    print("\n" + "="*40)
    print("         UNIFORM / SPACE-FILLING RESULT")
    print("="*40)
    print(f"Method                : {args.method}")
    print(f"Final Network Size    : {len(selected_sorted)} sensors")
    print(f"D-Optimal Objective   : {score:.6e}")
    print("="*40 + "\n")

    h5_file.close()

if __name__ == "__main__":
    main()