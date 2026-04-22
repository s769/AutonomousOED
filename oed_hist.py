import h5py
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import sys
import re
from mpi4py import MPI
from tqdm import tqdm
import seaborn as sns

# --- Standardized Styling ---
sns.set_context("paper", font_scale=1.3)
sns.set_style("whitegrid")
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


# --- Helper Functions ---

def get_block_from_h5(h5_dset, Nt, block_row_idx, block_col_idx, device, dtype):
    block_start_row, block_start_col = int(block_row_idx * Nt), int(block_col_idx * Nt)
    block_end_row, block_end_col = int(block_start_row + Nt), int(block_start_col + Nt)
    block_numpy = h5_dset[block_start_row:block_end_row, block_start_col:block_end_col]

    if np.isnan(block_numpy).any():
        block_numpy = np.nan_to_num(block_numpy, nan=0.0)

    return torch.from_numpy(block_numpy).to(device=device, dtype=dtype)


def build_k_submatrix_from_h5(h5_dset, Nt, S_indices, device, r_sq, dtype):
    current_size = len(S_indices) * Nt
    K_S = torch.zeros((current_size, current_size), dtype=dtype, device=device)

    for row_idx, s_row in enumerate(S_indices):
        for col_idx in range(row_idx, len(S_indices)):
            s_col = S_indices[col_idx]

            fetch_row, fetch_col = min(s_row, s_col), max(s_row, s_col)
            block = get_block_from_h5(
                h5_dset, Nt, fetch_row, fetch_col, device, dtype
            )

            if s_row > s_col:
                block = block.T

            scaled_block = r_sq * block
            actual_h, actual_w = scaled_block.shape
            r_start, c_start = row_idx * Nt, col_idx * Nt

            K_S[r_start : r_start + actual_h, c_start : c_start + actual_w] = (
                scaled_block
            )

            if col_idx > row_idx:
                K_S[c_start : c_start + actual_w, r_start : r_start + actual_h] = (
                    scaled_block.T
                )

    K_S.diagonal().add_(1.0)
    return K_S


def evaluate_config(config, h5_k, Nt, r_sq, device, dtype):
    K_S = build_k_submatrix_from_h5(h5_k, Nt, config, device, r_sq, dtype)

    max_diag = torch.max(torch.abs(K_S.diagonal())).item()
    if max_diag == 0:
        max_diag = 1.0

    L_S = None
    bad_sensor_idx = -1

    for jitter_factor in [0.0]:
        try:
            if jitter_factor > 0:
                K_S_temp = K_S.clone()
                K_S_temp.diagonal().add_(jitter_factor * max_diag)
                L_S = torch.linalg.cholesky(K_S_temp)
                del K_S_temp
            else:
                L_S = torch.linalg.cholesky(K_S)
            break
        except torch.linalg.LinAlgError as e:
            # --- SMART RETRY PARSING ---
            error_msg = str(e)
            match = re.search(r"leading minor of order (\d+)", error_msg)
            if match:
                minor_order = int(match.group(1))
                # Map the failing matrix minor back to the sensor index in the config array
                bad_sensor_idx = (minor_order - 1) // Nt
            continue

    del K_S
    torch.cuda.empty_cache()

    if L_S is None:
        return -np.inf, bad_sensor_idx

    # Standard D-Optimal Objective (Log-Determinant)
    reg_obj = 2 * torch.sum(torch.log(torch.diag(L_S))).item()

    del L_S
    torch.cuda.empty_cache()

    return reg_obj, None


def get_truncated_config(filepath, budget):
    data = np.loadtxt(filepath)[:].astype(float)
    indices = data[:, 0].astype(int)
    if len(indices) < budget:
        print(
            f"Warning: File {filepath} only has {len(indices)} sensors. Budget is {budget}."
        )
        return indices.tolist(), data[-1, 1]

    # Assuming column index 1 contains the standard D-optimal objective value
    return indices[:budget].tolist(), data[budget - 1, 1]


def plot_histogram(checkpoint_file, optimal_file, budget, args):
    print("Evaluating optimal configuration and generating plot...")

    if not os.path.exists(checkpoint_file):
        print(f"Error: {checkpoint_file} not found.")
        return

    rand_data = np.loadtxt(checkpoint_file)
    if rand_data.ndim > 1:
        # Fallback if an old 2-column checkpoint file is loaded
        rand_data = rand_data[:, 0]

    valid_reg = rand_data[~np.isinf(rand_data)]

    _, opt_reg_obj = get_truncated_config(optimal_file, budget)

    sns.set_palette("colorblind")
    color_hist = "silver"
    color_opt = sns.color_palette("colorblind")[0]

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

    if len(valid_reg) > 0:
        ax.hist(
            valid_reg,
            bins=50,
            color=color_hist,
            alpha=0.7,
            edgecolor="black",
            label="Random Configurations",
            density=True,
        )
        
    if not np.isinf(opt_reg_obj):
        ax.axvline(
            opt_reg_obj,
            color=color_opt,
            linestyle="dashed",
            linewidth=3,
            label="Optimal Configuration",
        )

    ax.set_xlabel("D-Optimal Objective Value")
    ax.set_ylabel("Probability Density")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.legend(loc="best", frameon=True, shadow=True)

    y_max = ax.get_ylim()[1]
    ax.set_ylim(0, y_max * 1.25)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "oed_histogram_standard.pdf"
    plt.savefig(plot_path)
    print(f"Histogram saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Histogram of Standard D-Optimal Objectives for Random Configurations"
    )

    parser.add_argument(
        "--h5_store_K", type=str, required=True, help="Path to HDF5 store for K."
    )
    parser.add_argument(
        "--optimal_file",
        type=str,
        required=False,
        help="Text file with optimal sensor indices.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=175,
        help="Number of sensors to pick (B). Default is 175.",
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=100,
        help="Total random samples to collect.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument("--r_sq", type=float, default=1.0, help="Scaling factor r^2.")
    parser.add_argument(
        "--precision", type=str, choices=["single", "double"], default="single"
    )
    parser.add_argument(
        "--exclude_indices",
        type=str,
        default="",
        help="Comma-separated sensor indices to exclude from sampling (default: none).",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="random_samples.txt",
        help="File to append/load results.",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip computation, just plot existing checkpoint file.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip plotting at the end (useful for generating checkpoint data only).",
    )

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        num_gpus = torch.cuda.device_count()
        local_rank = int(
            os.environ.get(
                "SLURM_LOCALID", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank)
            )
        )
        device_id = local_rank % num_gpus
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")

    torch.set_grad_enabled(False)
    compute_dtype = torch.float64 if args.precision == "double" else torch.float32

    if args.plot_only:
        if rank == 0:
            plot_histogram(
                args.checkpoint_file,
                args.optimal_file,
                args.budget,
                args,
            )
        return

    # --- HDF5 POSIX I/O Initialization ---
    try:
        h5_file_K = h5py.File(args.h5_store_K, mode="r", rdcc_nbytes=0)
        dset_name_K = "K_matrix" if "K_matrix" in h5_file_K else list(h5_file_K.keys())[0]
        h5_k = h5_file_K[dset_name_K]

        Nt = h5_k.chunks[0]
        Nd = h5_k.shape[0] // Nt
    except Exception as e:
        if rank == 0:
            print(f"Error opening HDF5 K matrix: {e}")
        comm.Abort()
        return

    exclude = []
    if args.exclude_indices.strip():
        exclude = [int(x) for x in args.exclude_indices.split(",") if x.strip()]
    available_indices = [i for i in range(Nd) if i not in set(exclude)]

    if rank == 0:
        print(f"Total available sensors (Nd): {Nd}")

        existing_samples = 0
        if os.path.exists(args.checkpoint_file):
            try:
                data = np.loadtxt(args.checkpoint_file)
                if data.ndim > 0:
                    existing_samples = len(data) if data.ndim > 1 else len(np.atleast_1d(data))
                print(
                    f"Found {existing_samples} existing valid samples in {args.checkpoint_file}."
                )
            except Exception:
                print("Could not read checkpoint file. Starting fresh.")

        samples_needed = max(0, args.total_samples - existing_samples)
        print(
            f"Generating {samples_needed} new random configurations (Budget = {args.budget})..."
        )

        rng = np.random.RandomState(args.seed + existing_samples)

        all_configs = [
            rng.choice(available_indices, args.budget, replace=False).tolist()
            for _ in range(samples_needed)
        ]

        chunks = [all_configs[i::size] for i in range(size)]
    else:
        chunks = None

    local_configs = comm.scatter(chunks, root=0)
    local_results = []
    local_retries = 0

    worker_rng = np.random.RandomState(args.seed + 100000 + rank)
    rank_checkpoint = f"{args.checkpoint_file}.rank{rank}"

    iterator = (
        tqdm(local_configs, desc=f"Rank {rank} Computing")
        if rank == 0
        else local_configs
    )

    try:
        for config in iterator:
            reg_obj, bad_idx = evaluate_config(
                config, h5_k, Nt, args.r_sq, device, compute_dtype
            )

            # --- SMART RETRY LOOP ---
            while np.isinf(reg_obj):
                local_retries += 1

                if bad_idx is not None and 0 <= bad_idx < len(config):
                    old_sensor = config[bad_idx]
                    available_for_replacement = list(set(available_indices) - set(config))
                    new_sensor = worker_rng.choice(available_for_replacement)

                    print(
                        f"[Retry] Rank {rank}: Sensor {old_sensor} caused ill-conditioning. Replacing with {new_sensor}..."
                    )
                    config[bad_idx] = new_sensor
                else:
                    print(
                        f"[Retry] Rank {rank}: Unparseable ill-conditioning. Generating an entirely new configuration..."
                    )
                    config = worker_rng.choice(
                        available_indices, args.budget, replace=False
                    ).tolist()

                reg_obj, bad_idx = evaluate_config(
                    config, h5_k, Nt, args.r_sq, device, compute_dtype
                )

            local_results.append(reg_obj)

            with open(rank_checkpoint, "a") as f:
                np.savetxt(f, [[reg_obj]], fmt="%.6e")

        all_results = comm.gather(local_results, root=0)
        total_retries = comm.reduce(local_retries, op=MPI.SUM, root=0)

        if rank == 0:
            if len(local_configs) > 0:
                flat_results = [item for sublist in all_results for item in sublist]
                flat_results_np = np.array(flat_results)

                with open(args.checkpoint_file, "a") as f:
                    np.savetxt(f, flat_results_np, fmt="%.6e")

                print("\n" + "=" * 40)
                print("        GENERATION REPORT")
                print("=" * 40)
                print(f"Target Valid Samples: {len(flat_results)}")
                print(f"Total Retries Issued: {total_retries}")
                print(f"Master Checkpoint:    {args.checkpoint_file}")
                print("=" * 40 + "\n")

            if not args.no_plot:
                if not args.optimal_file:
                    raise ValueError("--optimal_file is required unless --no_plot is set.")
                plot_histogram(
                    args.checkpoint_file,
                    args.optimal_file,
                    args.budget,
                    args,
                )
            
    finally:
        if 'h5_file_K' in locals():
            h5_file_K.close()


if __name__ == "__main__":
    main()