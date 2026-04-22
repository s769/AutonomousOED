import h5py
import torch
import numpy as np
from mpi4py import MPI
from tqdm import tqdm
import argparse
import math
import os


def load_block_into_tensor(
    h5_dset,
    Nt,
    row_idx,
    col_idx,
    destination_tensor,
    dest_row_start,
    dest_col_start,
    dtype,
):
    block_start_row, block_start_col = row_idx * Nt, col_idx * Nt
    block_numpy = h5_dset[
        block_start_row : block_start_row + Nt, block_start_col : block_start_col + Nt
    ]
    destination_tensor[
        dest_row_start : dest_row_start + Nt, dest_col_start : dest_col_start + Nt
    ] = torch.from_numpy(block_numpy).to(dtype)


def fill_column_buffer(h5_dset, Nt, S_indices, candidate_idx, buffer_tensor, dtype):
    for i, s_row in enumerate(S_indices):
        load_block_into_tensor(
            h5_dset,
            Nt,
            row_idx=s_row,
            col_idx=candidate_idx,
            destination_tensor=buffer_tensor,
            dest_row_start=i * Nt,
            dest_col_start=0,
            dtype=dtype,
        )


def build_k_submatrix_from_h5(h5_dset, Nt, S_indices, device, r_sq, dtype):
    current_size = len(S_indices) * Nt
    K_S = torch.zeros((current_size, current_size), dtype=dtype, device=device)

    for row_idx, s_row in enumerate(S_indices):
        for col_idx, s_col in enumerate(S_indices):
            if col_idx >= row_idx:
                block_start_row, block_start_col = s_row * Nt, s_col * Nt
                block_numpy = h5_dset[
                    block_start_row : block_start_row + Nt,
                    block_start_col : block_start_col + Nt,
                ]
                block_gpu = torch.from_numpy(block_numpy).to(device, dtype=dtype)
                block_gpu.mul_(r_sq)

                K_S[
                    row_idx * Nt : (row_idx + 1) * Nt, col_idx * Nt : (col_idx + 1) * Nt
                ] = block_gpu
                if col_idx > row_idx:
                    K_S[
                        col_idx * Nt : (col_idx + 1) * Nt,
                        row_idx * Nt : (row_idx + 1) * Nt,
                    ] = block_gpu.T

    K_S.diagonal().add_(1.0)
    return K_S


def main():
    parser = argparse.ArgumentParser(
        description="Low-Memory HDF5-Optimized Parallel Greedy Selection (Pipelined)"
    )
    parser.add_argument("h5_store", type=str, help="Path to 2D Chunked HDF5 file.")
    parser.add_argument("budget", type=int, help="Number of sensors to select.")
    parser.add_argument("--r_sq", type=float, required=True, help="Scaling factor.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("--checkpoint_file", type=str, default="checkpoint.txt")
    parser.add_argument("--restart_from", type=str, default=None)
    parser.add_argument(
        "--precision", type=str, choices=["single", "double"], default="double"
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")

    torch.set_grad_enabled(False)
    compute_dtype = torch.float64 if args.precision == "double" else torch.float32

    # --- HDF5 POSIX I/O Initialization (Removed MPI Driver) ---
    try:
        h5_file = h5py.File(args.h5_store, mode="r", rdcc_nbytes=0)
        dset_name = "K_matrix" if "K_matrix" in h5_file else list(h5_file.keys())[0]
        h5_dset = h5_file[dset_name]

        N, _ = h5_dset.shape
        Nt = h5_dset.chunks[0]
        Nd = N // Nt

        h5_dtype = h5_dset.dtype
        torch_h5_dtype = torch.float64 if h5_dtype == np.float64 else torch.float32

    except Exception as e:
        if rank == 0:
            print(f"Error opening HDF5: {e}")
        comm.Abort()
        return

    all_indices = list(range(Nd))
    candidates_per_worker = math.ceil(Nd / size)
    start_idx = rank * candidates_per_worker
    end_idx = min((rank + 1) * candidates_per_worker, Nd)
    local_candidate_indices = all_indices[start_idx:end_idx]

    S_indices = []
    log_dets = []
    current_log_det = 0.0

    # --- GLOBAL PRE-ALLOCATION ---
    max_size = args.budget * Nt
    L_S_global = torch.zeros((max_size, max_size), device=device, dtype=compute_dtype)
    L_S = L_S_global[0:0, 0:0]

    # --- PIPELINE MEMORY PRE-ALLOCATION (2-STAGE) ---
    needs_cast = torch_h5_dtype != compute_dtype

    pinned_Si = [
        torch.empty((max_size, Nt), dtype=torch_h5_dtype).pin_memory() for _ in range(2)
    ]
    pinned_ii = [
        torch.empty((Nt, Nt), dtype=torch_h5_dtype).pin_memory() for _ in range(2)
    ]

    K_Si_gpu_raw = [
        torch.empty((max_size, Nt), dtype=torch_h5_dtype, device=device)
        for _ in range(2)
    ]
    K_ii_gpu_raw = [
        torch.empty((Nt, Nt), dtype=torch_h5_dtype, device=device) for _ in range(2)
    ]

    K_Si_gpu_math = (
        [
            torch.empty((max_size, Nt), dtype=compute_dtype, device=device)
            for _ in range(2)
        ]
        if needs_cast
        else K_Si_gpu_raw
    )
    K_ii_gpu_math = (
        [torch.empty((Nt, Nt), dtype=compute_dtype, device=device) for _ in range(2)]
        if needs_cast
        else K_ii_gpu_raw
    )

    # Pre-allocate Cholesky out buffers
    dummy_info = [torch.empty((), dtype=torch.int32, device=device) for _ in range(2)]

    # Explicit CUDA streams for overlap
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]

    gpu_done_events = [torch.cuda.Event() for _ in range(2)]
    for e in gpu_done_events:
        e.record(torch.cuda.default_stream)

    if rank == 0:
        print(f"--- Low-Memory Pipelined Selection (Rank 0 on {device}) ---")
        if args.restart_from:
            try:
                loaded = np.loadtxt(args.restart_from)
                if loaded.ndim == 1:
                    loaded = loaded.reshape(1, -1)
                S_indices = loaded[:, 0].astype(int).tolist()
                log_dets = loaded[:, 1].tolist()
                print(f"Loaded {len(S_indices)} sensors.")
            except Exception as e:
                print(f"Restart failed: {e}. Starting fresh.")
                S_indices = []

    S_indices = comm.bcast(S_indices, root=0)

    if len(S_indices) > 0:
        if rank == 0:
            print("Rebuilding Cholesky factor...")
        K_S_rebuilt = build_k_submatrix_from_h5(
            h5_dset, Nt, S_indices, device, args.r_sq, compute_dtype
        )
        torch.linalg.cholesky(K_S_rebuilt, out=K_S_rebuilt)

        current_k_size = len(S_indices) * Nt
        L_S_global[:current_k_size, :current_k_size] = K_S_rebuilt
        L_S = L_S_global[:current_k_size, :current_k_size]

        current_log_det = 2 * torch.sum(torch.log(torch.diag(L_S)))
        del K_S_rebuilt
        torch.cuda.empty_cache()

    if rank == 0:
        pbar = tqdm(total=args.budget, initial=len(S_indices), desc="Selecting")

    try:
        for k in range(len(S_indices), args.budget):
            valid_candidates = [
                i for i in local_candidate_indices if i not in S_indices
            ]
            actual_evals = len(valid_candidates)

            # Double-buffered trackers to prevent stream race conditions
            best_tracker_gpu = [
                torch.tensor([-float("inf"), -1.0], dtype=compute_dtype, device=device)
                for _ in range(2)
            ]
            failure_count_gpu = [
                torch.tensor([0], dtype=torch.int32, device=device) for _ in range(2)
            ]

            def load_candidate_to_buffer(c_idx, buf_idx):
                i = valid_candidates[c_idx]
                col_start = i * Nt
                col_end = col_start + Nt

                np_Si_pinned_view = pinned_Si[buf_idx].numpy()
                np_ii_pinned_view = pinned_ii[buf_idx].numpy()

                if k > 0:
                    for idx, s_row in enumerate(S_indices):
                        row_start = s_row * Nt
                        row_end = row_start + Nt
                        np_Si_pinned_view[idx * Nt : (idx + 1) * Nt, :] = h5_dset[
                            row_start:row_end, col_start:col_end
                        ]

                np_ii_pinned_view[:, :] = h5_dset[col_start:col_end, col_start:col_end]

            # ------------------------------------------------
            # PRE-FILL PIPELINE
            # ------------------------------------------------
            if actual_evals > 0:
                load_candidate_to_buffer(0, 0)

            for c, i in enumerate(valid_candidates):
                curr_b = c % 2
                next_b = (c + 1) % 2

                # Slice buffers to current active K size
                curr_Si_raw = K_Si_gpu_raw[curr_b][: k * Nt, :] if k > 0 else None
                curr_Si_math = K_Si_gpu_math[curr_b][: k * Nt, :] if k > 0 else None

                # ------------------------------------------------
                # ASYNC GPU PIPELINE
                # ------------------------------------------------
                with torch.cuda.stream(streams[curr_b]):
                    if k > 0:
                        curr_Si_raw.copy_(
                            pinned_Si[curr_b][: k * Nt, :], non_blocking=True
                        )
                    K_ii_gpu_raw[curr_b].copy_(pinned_ii[curr_b], non_blocking=True)

                    if needs_cast:
                        if k > 0:
                            curr_Si_math.copy_(curr_Si_raw, non_blocking=True)
                        K_ii_gpu_math[curr_b].copy_(
                            K_ii_gpu_raw[curr_b], non_blocking=True
                        )

                    if k > 0:
                        curr_Si_math.mul_(args.r_sq)
                        # Strict In-Place Solve
                        torch.linalg.solve_triangular(
                            L_S, curr_Si_math, upper=False, out=curr_Si_math
                        )

                    K_ii_gpu_math[curr_b].mul_(args.r_sq).diagonal().add_(1.0)

                    if k > 0:
                        # Strict In-Place Schur
                        K_ii_gpu_math[curr_b].addmm_(
                            curr_Si_math.T, curr_Si_math, beta=1.0, alpha=-1.0
                        )

                    torch.linalg.cholesky_ex(
                        K_ii_gpu_math[curr_b],
                        check_errors=False,
                        out=(K_ii_gpu_math[curr_b], dummy_info[curr_b]),
                    )

                    current_score = 2 * torch.sum(
                        torch.log(torch.diag(K_ii_gpu_math[curr_b]))
                    )

                    is_better = current_score > best_tracker_gpu[curr_b][0]
                    best_tracker_gpu[curr_b][0] = torch.where(
                        is_better, current_score, best_tracker_gpu[curr_b][0]
                    )
                    best_tracker_gpu[curr_b][1] = torch.where(
                        is_better,
                        torch.tensor(float(i), device=device, dtype=compute_dtype),
                        best_tracker_gpu[curr_b][1],
                    )

                    failure_count_gpu[curr_b] += (dummy_info[curr_b] > 0).to(
                        torch.int32
                    )

                    gpu_done_events[curr_b].record(streams[curr_b])

                # ------------------------------------------------
                # OVERLAPPED CPU I/O
                # ------------------------------------------------
                if c + 1 < actual_evals:
                    gpu_done_events[next_b].synchronize()
                    load_candidate_to_buffer(c + 1, next_b)

            # Flush pipeline before communication and score reduction
            torch.cuda.synchronize()

            # Reduce the two stream trackers to find local best
            score_0, cand_0 = (
                best_tracker_gpu[0][0].item(),
                best_tracker_gpu[0][1].item(),
            )
            score_1, cand_1 = (
                best_tracker_gpu[1][0].item(),
                best_tracker_gpu[1][1].item(),
            )

            if score_0 >= score_1:
                local_best_score = score_0
                local_best_candidate = int(cand_0)
            else:
                local_best_score = score_1
                local_best_candidate = int(cand_1)

            local_failures = failure_count_gpu[0].item() + failure_count_gpu[1].item()

            local_result = np.array(
                [local_best_score, local_best_candidate, local_failures], dtype="d"
            )

            global_results = np.empty([size, 3], dtype="d") if rank == 0 else None
            comm.Gather(local_result, global_results, root=0)

            best_candidate_idx = -1
            new_gain = 0.0

            if rank == 0:
                best_worker = np.argmax(global_results[:, 0])
                new_gain = global_results[best_worker, 0]
                best_candidate_idx = int(global_results[best_worker, 1])

                if args.verbose:
                    mem_used = (
                        torch.cuda.max_memory_allocated(device) / 1e9
                        if torch.cuda.is_available()
                        else 0
                    )
                    print(
                        f"Iter {k + 1}: Selected {best_candidate_idx} (Gain: {new_gain:.4f}, GPU Peak: {mem_used:.2f}GB)"
                    )

            best_candidate_idx = comm.bcast(best_candidate_idx, root=0)
            new_gain = comm.bcast(new_gain, root=0)

            if best_candidate_idx == -1:
                if rank == 0:
                    print("No valid candidate found.")
                break

            S_indices.append(best_candidate_idx)
            current_log_det += new_gain

            # --- ONE-TIME SYNCHRONOUS L_S UPDATE ---
            K_ii_new = torch.empty((Nt, Nt), dtype=compute_dtype, device=device)
            load_block_into_tensor(
                h5_dset,
                Nt,
                best_candidate_idx,
                best_candidate_idx,
                K_ii_new,
                0,
                0,
                compute_dtype,
            )
            K_ii_new.mul_(args.r_sq).diagonal().add_(1.0)

            if k == 0:
                torch.linalg.cholesky(K_ii_new, out=K_ii_new)
                L_S_global[0:Nt, 0:Nt] = K_ii_new
                L_S = L_S_global[0:Nt, 0:Nt]
            else:
                K_Si_new = torch.empty((k * Nt, Nt), dtype=compute_dtype, device=device)
                fill_column_buffer(
                    h5_dset,
                    Nt,
                    S_indices[:-1],
                    best_candidate_idx,
                    K_Si_new,
                    compute_dtype,
                )
                K_Si_new.mul_(args.r_sq)

                # Strict In-Place Math for Synchronous Update
                torch.linalg.solve_triangular(L_S, K_Si_new, upper=False, out=K_Si_new)
                K_ii_new.addmm_(K_Si_new.T, K_Si_new, beta=1.0, alpha=-1.0)
                torch.linalg.cholesky(K_ii_new, out=K_ii_new)

                L_S_global[k * Nt : (k + 1) * Nt, : k * Nt] = K_Si_new.T
                L_S_global[k * Nt : (k + 1) * Nt, k * Nt : (k + 1) * Nt] = K_ii_new

                L_S = L_S_global[: (k + 1) * Nt, : (k + 1) * Nt]
                del K_Si_new

            del K_ii_new

            if rank == 0:
                pbar.update(1)
                log_dets.append(
                    current_log_det.item()
                    if torch.is_tensor(current_log_det)
                    else current_log_det
                )

                if (k + 1) % 10 == 0:
                    save_arr = np.array(list(zip(S_indices, log_dets)), dtype="d")
                    np.savetxt(args.checkpoint_file, save_arr, fmt="%d %.18e")

    except KeyboardInterrupt:
        if rank == 0:
            print("\nInterrupted.")
    except torch.cuda.OutOfMemoryError:
        if rank == 0:
            print(
                "\nOOM Error. Try reducing precision (--precision single) or checking budget."
            )
    except Exception as e:
        if rank == 0:
            print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "h5_file" in locals():
            h5_file.close()

        if rank == 0:
            if "pbar" in locals():
                pbar.close()
            if len(S_indices) > 0:
                save_arr = np.array(list(zip(S_indices, log_dets)), dtype="d")
                np.savetxt(args.checkpoint_file, save_arr, fmt="%d %.18e")
                print(f"Saved {len(S_indices)} sensors to {args.checkpoint_file}")


if __name__ == "__main__":
    main()
