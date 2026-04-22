from mpi4py import MPI
import torch
import torch.cuda.nvtx as nvtx  # NVTX markers for Nsight Systems profiling
import argparse
import numpy as np
import os
import gc
import h5py
import math
from tqdm import tqdm


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_benchmark(filename, h5_path, total_candidates, Nt, k, runs, max_evals, r_sq, seed_s, seed_candidates):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if not torch.cuda.is_available():
        if rank == 0:
            print("CUDA is not available!")
        comm.Abort()

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    dtype = torch.float32

    local_candidates = total_candidates // size
    actual_evals = local_candidates
    if max_evals > 0:
        actual_evals = min(local_candidates, max_evals)

    if actual_evals == 0:
        if rank == 0:
            print("Error: No candidates to evaluate. Check scaling parameters.")
        comm.Abort()

    cleanup()

    # ---------------------------------------------------------
    # 1. OPEN HDF5 & DETECT NATIVE DTYPES (POSIX I/O)
    # ---------------------------------------------------------
    if rank == 0:
        print(f"Opening HDF5 store at: {h5_path}")
        print(
            f"Ranks: {size} | Local pool: {local_candidates} | Actually evaluating: {actual_evals}"
        )

    try:
        # Standard POSIX reads
        h5_file = h5py.File(h5_path, mode="r", rdcc_nbytes=0)
        dset_name = "K_matrix" if "K_matrix" in h5_file else list(h5_file.keys())[0]
        h5_dset = h5_file[dset_name]
        total_h5_cols = h5_dset.shape[1]
        Nd = total_h5_cols // Nt

        h5_dtype = h5_dset.dtype
        torch_h5_dtype = torch.float64 if h5_dtype == np.float64 else torch.float32

    except Exception as e:
        if rank == 0:
            print(f"Error opening HDF5: {e}")
        comm.Abort()

    # Setup zero-copy memory pipeline & streams
    current_size = k * Nt

    L_S_cpu = torch.eye(current_size, dtype=dtype) * math.sqrt(r_sq)
    L_S = L_S_cpu.to(device)

    needs_cast = torch_h5_dtype != dtype

    # Stage 1: Pinned CPU Memory (Source for H2D Transfers)
    pinned_Si = [
        torch.empty((current_size, Nt), dtype=torch_h5_dtype).pin_memory()
        for _ in range(2)
    ]
    pinned_ii = [
        torch.empty((Nt, Nt), dtype=torch_h5_dtype).pin_memory() for _ in range(2)
    ]

    # Stage 2: GPU Memory (Double Buffered)
    K_Si_gpu_raw = [
        torch.empty((current_size, Nt), dtype=torch_h5_dtype, device=device)
        for _ in range(2)
    ]
    K_ii_gpu_raw = [
        torch.empty((Nt, Nt), dtype=torch_h5_dtype, device=device) for _ in range(2)
    ]

    K_Si_gpu_math = (
        [torch.empty((current_size, Nt), dtype=dtype, device=device) for _ in range(2)]
        if needs_cast
        else K_Si_gpu_raw
    )
    K_ii_gpu_math = (
        [torch.empty((Nt, Nt), dtype=dtype, device=device) for _ in range(2)]
        if needs_cast
        else K_ii_gpu_raw
    )

    dummy_info = [torch.empty((), dtype=torch.int32, device=device) for _ in range(2)]

    # Setup isolated CUDA streams for overlap
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    gpu_done_events = [torch.cuda.Event() for _ in range(2)]
    for e in gpu_done_events:
        e.record()

    np.random.seed(seed_s)
    mock_S_indices = np.random.choice(Nd, k, replace=False)
    valid_pool = np.setdiff1d(np.arange(Nd), mock_S_indices)

    np.random.seed(seed_candidates)
    global_random_sequence = np.random.choice(
        valid_pool, size=total_candidates, replace=True
    )

    start_idx = rank * local_candidates
    local_random_sequence = global_random_sequence[start_idx : start_idx + actual_evals]

    def load_candidate_to_buffer(c_idx, buf_idx):
        t0 = MPI.Wtime()
        col_start = local_random_sequence[c_idx] * Nt
        col_end = col_start + Nt

        np_Si_pinned_view = pinned_Si[buf_idx].numpy()
        np_ii_pinned_view = pinned_ii[buf_idx].numpy()

        for i, s_row in enumerate(mock_S_indices):
            row_start = s_row * Nt
            row_end = row_start + Nt
            np_Si_pinned_view[i * Nt : (i + 1) * Nt, :] = h5_dset[
                row_start:row_end, col_start:col_end
            ]

        np_ii_pinned_view[:, :] = h5_dset[col_start:col_end, col_start:col_end]

        return MPI.Wtime() - t0

    global_best_io = float("inf")
    global_best_comp = float("inf")
    global_best_wall = float("inf")
    global_best_imbalance = float("inf")
    global_best_comm = float("inf")

    for r in range(runs):
        comm.Barrier()

        local_io_time = 0.0
        local_compute_time = 0.0

        # Split score trackers to prevent stream write collisions
        local_best_score_gpu = [
            torch.tensor([-float("inf")], dtype=dtype, device=device) for _ in range(2)
        ]

        t_wall_start = MPI.Wtime()

        # Fill the first pipeline buffer
        if actual_evals > 0:
            nvtx.range_push("Lustre_IO_Read")
            local_io_time += load_candidate_to_buffer(0, buf_idx=0)
            nvtx.range_pop()

        for c in tqdm(range(actual_evals), disable=rank != 0):
            curr_b = c % 2
            next_b = (c + 1) % 2

            nvtx.range_push(f"Candidate_Loop_{c}")

            # ------------------------------------------------
            # 1. ASYNC GPU PIPELINE (ISOLATED STREAM)
            # ------------------------------------------------
            with torch.cuda.stream(streams[curr_b]):
                nvtx.range_push("GPU_Dispatch")

                # H2D Transfers enqueue on this specific stream
                K_Si_gpu_raw[curr_b].copy_(pinned_Si[curr_b], non_blocking=True)
                K_ii_gpu_raw[curr_b].copy_(pinned_ii[curr_b], non_blocking=True)

                if needs_cast:
                    K_Si_gpu_math[curr_b].copy_(K_Si_gpu_raw[curr_b], non_blocking=True)
                    K_ii_gpu_math[curr_b].copy_(K_ii_gpu_raw[curr_b], non_blocking=True)

                start_events[curr_b].record(streams[curr_b])

                # Math: Strict In-Place Operations
                K_Si_gpu_math[curr_b].mul_(r_sq)
                torch.linalg.solve_triangular(
                    L_S, K_Si_gpu_math[curr_b], upper=False, out=K_Si_gpu_math[curr_b]
                )

                K_ii_gpu_math[curr_b].mul_(r_sq).diagonal().add_(1.0)
                K_ii_gpu_math[curr_b].addmm_(
                    K_Si_gpu_math[curr_b].T, K_Si_gpu_math[curr_b], alpha=-1.0, beta=1.0
                )

                torch.linalg.cholesky_ex(
                    K_ii_gpu_math[curr_b],
                    check_errors=False,
                    out=(K_ii_gpu_math[curr_b], dummy_info[curr_b]),
                )

                current_score = 2.0 * torch.sum(
                    torch.log(torch.diag(K_ii_gpu_math[curr_b]))
                )
                local_best_score_gpu[curr_b] = torch.max(
                    local_best_score_gpu[curr_b], current_score
                )

                end_events[curr_b].record(streams[curr_b])
                gpu_done_events[curr_b].record(streams[curr_b])
                nvtx.range_pop()

            # ------------------------------------------------
            # 2. OVERLAPPED CPU I/O
            # ------------------------------------------------
            if c + 1 < actual_evals:
                # CPU Wait: Ensure the *next* buffer's stream has finished its prior
                # iteration before the CPU overwrites its pinned memory.
                nvtx.range_push("CPU_Wait_For_GPU")
                gpu_done_events[next_b].synchronize()
                nvtx.range_pop()

                if c >= 1:
                    local_compute_time += (
                        start_events[next_b].elapsed_time(end_events[next_b]) / 1000.0
                    )

                nvtx.range_push("Lustre_IO_Read")
                local_io_time += load_candidate_to_buffer(c + 1, buf_idx=next_b)
                nvtx.range_pop()

            nvtx.range_pop()

        # Pipeline Drain: Sync all streams before final timing collection
        torch.cuda.synchronize()

        local_best_score = max(
            local_best_score_gpu[0].item(), local_best_score_gpu[1].item()
        )

        # Collect final timings that fell out of the loop
        if actual_evals >= 1:
            local_compute_time += (
                start_events[(actual_evals - 1) % 2].elapsed_time(
                    end_events[(actual_evals - 1) % 2]
                )
                / 1000.0
            )
        if actual_evals >= 2:
            local_compute_time += (
                start_events[(actual_evals - 2) % 2].elapsed_time(
                    end_events[(actual_evals - 2) % 2]
                )
                / 1000.0
            )

        local_wall_time = MPI.Wtime() - t_wall_start

        comm.Barrier()
        t_comm_start = MPI.Wtime()
        _ = comm.allreduce(local_best_score, op=MPI.MAX)
        t_comm_end = MPI.Wtime()

        local_comm_time = t_comm_end - t_comm_start

        max_io_time = comm.allreduce(local_io_time, op=MPI.MAX)
        sum_comp_time = comm.allreduce(local_compute_time, op=MPI.SUM)
        max_comp_time = comm.allreduce(local_compute_time, op=MPI.MAX)
        max_wall_time = comm.allreduce(local_wall_time, op=MPI.MAX)
        max_comm_time = comm.allreduce(local_comm_time, op=MPI.MAX)

        avg_comp_time = sum_comp_time / size
        run_imbalance = max_comp_time - avg_comp_time

        global_best_io = min(global_best_io, max_io_time)
        global_best_comp = min(global_best_comp, avg_comp_time)
        global_best_wall = min(global_best_wall, max_wall_time)
        global_best_imbalance = min(global_best_imbalance, run_imbalance)
        global_best_comm = min(global_best_comm, max_comm_time)

    h5_file.close()

    if rank == 0:
        file_exists = os.path.isfile(filename)
        with open(filename, "a") as f:
            if not file_exists:
                f.write(
                    "ranks,actual_evals,io_time,compute_time,wall_time,imbalance_time,comm_time\n"
                )
            f.write(
                f"{size},{actual_evals},{global_best_io:.6f},{global_best_comp:.6f},{global_best_wall:.6f},{global_best_imbalance:.6f},{global_best_comm:.6f}\n"
            )

        print(
            f"Ranks={size} | Evals/GPU={actual_evals} | Wall Time: {global_best_wall:.4f}s | MAX I/O: {global_best_io:.4f}s | MEAN Comp: {global_best_comp:.4f}s"
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="mpi_results_scaling.csv")
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--total_candidates", type=int, default=10000)
    parser.add_argument("--Nt", type=int, default=420)
    parser.add_argument("--k", type=int, default=175)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--r_sq", type=float, default=1e4, help="Scaling factor r^2.")
    parser.add_argument("--seed_s", type=int, default=100, help="Seed for initial selected-set mock indices.")
    parser.add_argument("--seed_candidates", type=int, default=42, help="Seed for candidate evaluation order.")
    parser.add_argument(
        "--max_evals",
        type=int,
        default=256,
        help="Cap the number of evals per rank to prevent excessive runtimes.",
    )
    args = parser.parse_args()

    run_benchmark(
        args.file,
        args.h5_path,
        args.total_candidates,
        args.Nt,
        args.k,
        args.runs,
        args.max_evals,
        args.r_sq,
        args.seed_s,
        args.seed_candidates,
    )
