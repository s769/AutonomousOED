import h5py
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
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


def fill_column_buffer(h5_dset, Nt, S_indices, candidate_idx, buffer_tensor, device, dtype):
    """Fetches the off-diagonal blocks K_{S, c} for the Schur update."""
    for i, s_row in enumerate(S_indices):
        fetch_row, fetch_col = min(s_row, candidate_idx), max(s_row, candidate_idx)
        block = get_block_from_h5(h5_dset, Nt, fetch_row, fetch_col, device, dtype)

        # Transpose if we had to fetch the upper triangle but need the lower
        if s_row > candidate_idx:
            block = block.T

        buffer_tensor[i * Nt : (i + 1) * Nt, :] = block


def build_random_config_sequentially(
    h5_dset, Nt, available_indices, budget, device, compute_dtype, r_sq, rng
):
    """
    Builds a purely random sensor configuration incrementally.
    Uses the Schur complement update to guarantee the matrix remains SPD.
    If a randomly selected sensor is highly collinear with the current set, it is rejected.
    """
    max_size = budget * Nt
    L_S_global = torch.zeros((max_size, max_size), device=device, dtype=compute_dtype)

    S_indices = []
    current_pool = list(available_indices)
    current_log_det = 0.0

    for k in range(budget):
        # Shuffle the available pool so we draw candidates in a random order
        rng.shuffle(current_pool)
        found_valid_sensor = False

        for c in current_pool:
            # 1. Fetch the diagonal block for the candidate
            K_cc = get_block_from_h5(h5_dset, Nt, c, c, device, compute_dtype)
            K_cc = 0.5 * (K_cc + K_cc.T)  # Ensure strict symmetry
            K_cc.mul_(r_sq).diagonal().add_(1.0)

            if k == 0:
                try:
                    L_cc = torch.linalg.cholesky(K_cc)
                    L_S_global[0:Nt, 0:Nt] = L_cc
                    current_log_det += 2 * torch.sum(torch.log(torch.diag(L_cc))).item()
                    found_valid_sensor = True
                except torch.linalg.LinAlgError:
                    pass
            else:
                # 2. Fetch the off-diagonal blocks K_{S, c}
                K_Sc = torch.empty((k * Nt, Nt), dtype=compute_dtype, device=device)
                fill_column_buffer(h5_dset, Nt, S_indices, c, K_Sc, device, compute_dtype)
                K_Sc.mul_(r_sq)

                L_S_current = L_S_global[: k * Nt, : k * Nt]

                # 3. Schur Complement Update
                # Solve: L_S * Y = K_Sc  => Y = L_S \ K_Sc
                Y = torch.linalg.solve_triangular(L_S_current, K_Sc, upper=False)

                # K_Schur = K_cc - Y^T * Y
                K_Schur = K_cc - Y.T @ Y

                max_diag = torch.max(torch.abs(K_Schur.diagonal())).item()
                if max_diag == 0: 
                    max_diag = 1.0

                # Try Cholesky on the Schur complement with standard jitter loop
                for jitter in [0.0]:
                    try:
                        if jitter > 0:
                            K_temp = K_Schur.clone()
                            K_temp.diagonal().add_(jitter * max_diag)
                            L_cc = torch.linalg.cholesky(K_temp)
                        else:
                            L_cc = torch.linalg.cholesky(K_Schur)

                        # Success! Update the global Cholesky factor
                        L_S_global[k * Nt : (k + 1) * Nt, : k * Nt] = Y.T
                        L_S_global[k * Nt : (k + 1) * Nt, k * Nt : (k + 1) * Nt] = L_cc
                        
                        # Accumulate objective
                        current_log_det += 2 * torch.sum(torch.log(torch.diag(L_cc))).item()
                        found_valid_sensor = True
                        break
                    except torch.linalg.LinAlgError:
                        continue

            if found_valid_sensor:
                S_indices.append(c)
                current_pool.remove(c)  # Prevent picking it again
                break  # Break inner loop, move to next k

        if not found_valid_sensor:
            # We exhausted the entire remaining pool without finding an independent sensor
            # This is mathematically possible if budget is extremely close to Nd
            return None, None

    return S_indices, current_log_det


def pre_screen_sensors(h5_k, Nt, indices, device, dtype):
    """Removes sensors that are inherently non-SPD due to numerical artifacts."""
    valid_sensors = []
    print("\nPre-screening sensors for local positive-definiteness...")
    
    for s in tqdm(indices, desc="Screening"):
        block = get_block_from_h5(h5_k, Nt, s, s, device, dtype)
        
        block = 0.5 * (block + block.T)
        block.diagonal().add_(1.0)
        
        try:
            torch.linalg.cholesky(block)
            valid_sensors.append(s)
        except torch.linalg.LinAlgError:
            pass
            
        del block
        torch.cuda.empty_cache()
        
    removed = len(indices) - len(valid_sensors)
    if removed > 0:
        print(f"Removed {removed} poison-pill sensors from candidate pool.")
    else:
        print("All sensors passed positive-definiteness check.")
        
    return valid_sensors


def _append_checkpoint_line(path, value):
    with open(path, "a") as f:
        np.savetxt(f, [[value]], fmt="%.6e")
        f.flush()
        os.fsync(f.fileno())


class RankCheckpointMerger:
    """Incrementally append new per-rank checkpoint lines into the master file."""

    def __init__(self, checkpoint_path, comm_size):
        self.checkpoint_path = checkpoint_path
        self.offsets = [0] * comm_size

    def _rank_path(self, rank):
        return f"{self.checkpoint_path}.rank{rank}"

    def merge_new(self):
        """Append unmerged per-rank checkpoint lines to the master, then truncate them."""
        for rank in range(len(self.offsets)):
            rank_path = self._rank_path(rank)
            if not os.path.exists(rank_path):
                continue
            file_size = os.path.getsize(rank_path)
            if file_size <= self.offsets[rank]:
                continue
            with open(rank_path, "r") as src, open(self.checkpoint_path, "a") as dst:
                src.seek(self.offsets[rank])
                dst.write(src.read())
                dst.flush()
                os.fsync(dst.fileno())
            open(rank_path, "w").close()
            self.offsets[rank] = 0


def get_truncated_config(filepath, budget):
    data = np.loadtxt(filepath)[:].astype(float)
    indices = data[:, 0].astype(int)
    if len(indices) < budget:
        print(f"Warning: File {filepath} only has {len(indices)} sensors. Budget is {budget}.")
        return indices.tolist(), data[-1, 1]
    return indices[:budget].tolist(), data[budget - 1, 1]


def plot_histogram(checkpoint_file, optimal_value, uniform_value, budget, args):
    print("Evaluating optimal configuration and generating plot...")

    if not os.path.exists(checkpoint_file):
        print(f"Error: {checkpoint_file} not found.")
        return

    rand_data = np.loadtxt(checkpoint_file)
    if rand_data.ndim > 1:
        rand_data = rand_data[:, 0]

    valid_reg = rand_data[~np.isinf(rand_data)]

    sns.set_palette("colorblind")
    color_hist = "silver"
    color_opt = sns.color_palette("colorblind")[0]
    color_uniform = sns.color_palette("colorblind")[1]

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
        
    if optimal_value is not None:
        ax.axvline(
            optimal_value,
            color=color_opt,
            linestyle="dashed",
            linewidth=3,
            label="Optimal Configuration",
        )
    if uniform_value is not None:
        ax.axvline(
            uniform_value,
            color=color_uniform,
            linestyle="dashed",
            linewidth=3,
            label="Uniform Configuration",
        )   
    ax.set_xlabel("D-Optimal Objective Value")
    ax.set_ylabel("Probability Density")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.legend(loc="best", frameon=True, shadow=True)

    y_max = ax.get_ylim()[1]
    ax.set_ylim(0, y_max * 1.25)

    if optimal_value is not None and len(valid_reg) >= 2:
        sample_mean = np.mean(valid_reg)
        sample_stdev = np.std(valid_reg, ddof=1)
        if sample_stdev > 0:
            line_y_mean = 0.45 * y_max * 1.25
            n_sigma = (optimal_value - sample_mean) / sample_stdev
            sign = "+" if n_sigma >= 0 else ""
            ax.plot(
                [sample_mean, optimal_value],
                [line_y_mean, line_y_mean],
                color="black",
                linestyle=":",
                linewidth=1.5,
                zorder=5,
            )
            ax.text(
                0.5 * (sample_mean + optimal_value),
                line_y_mean,
                f"{sign}{n_sigma:.2f}$\\sigma$",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
                fontweight="bold",
            )

            sample_max = np.max(valid_reg)
            if sample_max != optimal_value:
                line_y_max = 0.60 * y_max * 1.25
                n_sigma_max = (optimal_value - sample_max) / sample_stdev
                sign_max = "+" if n_sigma_max >= 0 else ""
                ax.plot(
                    [sample_max, optimal_value],
                    [line_y_max, line_y_max],
                    color="black",
                    linestyle=":",
                    linewidth=1.5,
                    zorder=5,
                )
                ax.text(
                    0.5 * (sample_max + optimal_value),
                    line_y_max,
                    f"{sign_max}{n_sigma_max:.2f}$\\sigma$",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="black",
                    fontweight="bold",
                )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "oed_histogram_standard.pdf"
    plt.savefig(plot_path)
    print(f"Histogram saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Histogram using Sequential Block-Cholesky Random Sampling"
    )

    parser.add_argument("--h5_store_K", type=str, required=False, help="Path to HDF5 store for K.")
    parser.add_argument("--optimal_value", type=float, required=False, help="Optimal value of the D-optimal objective.")
    parser.add_argument("--budget", type=int, default=175, help="Number of sensors to pick (B). Default is 175.")
    parser.add_argument("--total_samples", type=int, default=100, help="Total random samples to collect.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--r_sq", type=float, default=1.0, help="Scaling factor r^2.")
    parser.add_argument("--precision", type=str, choices=["single", "double"], default="single")
    parser.add_argument("--exclude_indices", type=str, default="", help="Comma-separated sensor indices to exclude.")
    parser.add_argument("--checkpoint_file", type=str, default="random_samples.txt", help="File to append/load results.")
    parser.add_argument("--plot_only", action="store_true", help="Skip computation, just plot existing checkpoint file.")
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting at the end.")
    parser.add_argument("--uniform_value", type=float, required=False, help="Uniform grid value of the D-optimal objective.")

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
            plot_histogram(args.checkpoint_file, args.optimal_value, args.uniform_value, args.budget, args)
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
        available_indices = pre_screen_sensors(h5_k, Nt, available_indices, device, compute_dtype)
    else:
        available_indices = None
        
    available_indices = comm.bcast(available_indices, root=0)

    # Determine how many workloads each MPI worker needs to process
    local_samples_needed = 0
    checkpoint_merger = None
    if rank == 0:
        checkpoint_merger = RankCheckpointMerger(args.checkpoint_file, size)
        checkpoint_merger.merge_new()

        existing_samples = 0
        if os.path.exists(args.checkpoint_file):
            try:
                data = np.loadtxt(args.checkpoint_file)
                if data.ndim > 0:
                    existing_samples = len(data) if data.ndim > 1 else len(np.atleast_1d(data))
                print(f"Found {existing_samples} existing valid samples in {args.checkpoint_file}.")
            except Exception:
                print("Could not read checkpoint file. Starting fresh.")

        samples_needed = max(0, args.total_samples - existing_samples)
        print(f"Generating {samples_needed} new configurations (Budget = {args.budget})...")

        # Distribute the exact number of configurations to build to each rank
        chunks = [samples_needed // size + (1 if i < samples_needed % size else 0) for i in range(size)]
    else:
        chunks = None

    local_samples_needed = comm.scatter(chunks, root=0)
    local_results = []
    local_cornered_retries = 0

    worker_rng = np.random.RandomState(args.seed + rank * 1000)
    rank_checkpoint = f"{args.checkpoint_file}.rank{rank}"

    iterator = tqdm(range(local_samples_needed), desc=f"Rank {rank} Computing") if rank == 0 else range(local_samples_needed)

    try:
        for _ in iterator:
            while True:
                config, reg_obj = build_random_config_sequentially(
                    h5_k, Nt, available_indices, args.budget, device, compute_dtype, args.r_sq, worker_rng
                )

                if config is not None and not np.isinf(reg_obj):
                    local_results.append(reg_obj)
                    _append_checkpoint_line(rank_checkpoint, reg_obj)
                    if rank == 0:
                        checkpoint_merger.merge_new()
                    break
                else:
                    local_cornered_retries += 1
                    print(f"[Retry] Rank {rank}: Cornered! Restarting configuration buildup...")

        all_results = comm.gather(local_results, root=0)
        total_retries = comm.reduce(local_cornered_retries, op=MPI.SUM, root=0)

        if rank == 0:
            checkpoint_merger.merge_new()
            flat_results = [item for sublist in all_results for item in sublist]
            if flat_results:
                print("\n" + "=" * 40)
                print("        GENERATION REPORT")
                print("=" * 40)
                print(f"Target Valid Samples: {len(flat_results)}")
                print(f"Total Retries Issued: {total_retries}")
                print(f"Master Checkpoint:    {args.checkpoint_file}")
                print("=" * 40 + "\n")

            if not args.no_plot:
                if not args.optimal_value:
                    raise ValueError("--optimal_value is required unless --no_plot is set.")
                plot_histogram(args.checkpoint_file, args.optimal_value, args.budget, args)
            
    finally:
        if 'h5_file_K' in locals():
            h5_file_K.close()


if __name__ == "__main__":
    main()