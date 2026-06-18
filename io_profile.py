import torch
import numpy as np
import argparse
import time
import os
import h5py
import math
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

from h5_batch_io import build_si_read_plan, load_ii_block, load_si_column_blocks
from pipelined_loader import PipelinedLoader

def run_profiler(h5_path, Nt, k, max_evals, r_sq, precision, trace_file):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    
    compute_dtype = torch.float64 if precision == "double" else torch.float32
    element_size = 8 if precision == "double" else 4

    # ---------------------------------------------------------
    # 1. OPEN HDF5
    # ---------------------------------------------------------
    print(f"Opening HDF5 store at: {h5_path}")
    h5_file = h5py.File(h5_path, mode="r", rdcc_nbytes=0)
    dset_name = "K_matrix" if "K_matrix" in h5_file else list(h5_file.keys())[0]
    h5_dset = h5_file[dset_name]
    Nd = h5_dset.shape[0] // Nt

    h5_dtype = h5_dset.dtype
    torch_h5_dtype = torch.float64 if h5_dtype == np.float64 else torch.float32
    needs_cast = torch_h5_dtype != compute_dtype
    h5_element_size = 8 if h5_dtype == np.float64 else 4

    # ---------------------------------------------------------
    # 2. SETUP ZERO-COPY PIPELINE
    # ---------------------------------------------------------
    current_size = k * Nt
    L_S = (torch.eye(current_size, dtype=compute_dtype) * math.sqrt(r_sq)).to(device)

    pinned_Si = [torch.empty((current_size, Nt), dtype=torch_h5_dtype).pin_memory() for _ in range(2)]
    pinned_ii = [torch.empty((Nt, Nt), dtype=torch_h5_dtype).pin_memory() for _ in range(2)]

    K_Si_gpu_raw = [torch.empty((current_size, Nt), dtype=torch_h5_dtype, device=device) for _ in range(2)]
    K_ii_gpu_raw = [torch.empty((Nt, Nt), dtype=torch_h5_dtype, device=device) for _ in range(2)]

    K_Si_gpu_math = [torch.empty((current_size, Nt), dtype=compute_dtype, device=device) for _ in range(2)] if needs_cast else K_Si_gpu_raw
    K_ii_gpu_math = [torch.empty((Nt, Nt), dtype=compute_dtype, device=device) for _ in range(2)] if needs_cast else K_ii_gpu_raw

    dummy_info = [torch.empty((), dtype=torch.int32, device=device) for _ in range(2)]
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    gpu_done_events = [torch.cuda.Event() for _ in range(2)]
    for e in gpu_done_events: e.record()

    # Generate mock sequence
    np.random.seed(100)
    mock_S_indices = np.random.choice(Nd, k, replace=False)
    si_read_plan = build_si_read_plan(mock_S_indices, Nt)
    max_si_rows = max(
        (entry["h5_row_end"] - entry["h5_row_start"] for entry in si_read_plan),
        default=Nt,
    )
    si_slab = torch.empty((max_si_rows, Nt), dtype=torch_h5_dtype).pin_memory()
    si_slab_np = si_slab.numpy()
    print(
        f"Batched HDF5 reads: {len(si_read_plan)} hyperslabs for {k} sensor rows "
        f"(was {k} per candidate)"
    )
    valid_pool = np.setdiff1d(np.arange(Nd), mock_S_indices)
    candidate_sequence = np.random.choice(valid_pool, size=max_evals, replace=False)

    total_bytes_read = 0

    bytes_per_candidate = (k * Nt * Nt + Nt * Nt) * h5_element_size

    def load_candidate_to_buffer(c_idx, buf_idx):
        nonlocal total_bytes_read

        t0 = time.perf_counter()
        col_start = candidate_sequence[c_idx] * Nt
        col_end = col_start + Nt

        with record_function("POSIX_Lustre_Read"):
            np_Si_pinned_view = pinned_Si[buf_idx].numpy()
            np_ii_pinned_view = pinned_ii[buf_idx].numpy()
            load_si_column_blocks(
                h5_dset,
                col_start,
                col_end,
                Nt,
                np_Si_pinned_view,
                si_read_plan,
                slab_view=si_slab_np,
            )
            load_ii_block(h5_dset, col_start, col_end, np_ii_pinned_view)
        total_bytes_read += bytes_per_candidate
        return time.perf_counter() - t0

    loader = PipelinedLoader(load_candidate_to_buffer, cuda_device=device)
    loader.load_sync(0, 0)
    torch.cuda.synchronize()

    # ---------------------------------------------------------
    # 3. PROFILING LOOP
    # ---------------------------------------------------------
    print(f"\nStarting Profiling Loop for {max_evals} candidates...")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False
    ) as prof:
        
        for c in tqdm(range(max_evals)):
            curr_b = c % 2
            next_b = (c + 1) % 2

            with record_function(f"Candidate_Eval_{c}"):
                if c > 0:
                    with record_function("Wait_For_IO"):
                        loader.wait()

                if c + 1 < max_evals:
                    def _sync_next_buffer(event=gpu_done_events[next_b]):
                        with record_function("IO_Wait_For_GPU"):
                            event.synchronize()

                    loader.start_async(c + 1, next_b, before_load=_sync_next_buffer)

                with record_function("GPU_Stream_Dispatch"):
                    with torch.cuda.stream(streams[curr_b]):
                        with record_function("H2D_Transfer"):
                            K_Si_gpu_raw[curr_b].copy_(pinned_Si[curr_b], non_blocking=True)
                            K_ii_gpu_raw[curr_b].copy_(pinned_ii[curr_b], non_blocking=True)

                        with record_function("Math_Operations"):
                            if needs_cast:
                                K_Si_gpu_math[curr_b].copy_(K_Si_gpu_raw[curr_b], non_blocking=True)
                                K_ii_gpu_math[curr_b].copy_(K_ii_gpu_raw[curr_b], non_blocking=True)

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

                        gpu_done_events[curr_b].record(streams[curr_b])

        loader.wait()
        torch.cuda.synchronize()

    # ---------------------------------------------------------
    # 4. EXPORT & BANDWIDTH CALCULATION
    # ---------------------------------------------------------
    h5_file.close()
    
    print("\nExporting Chrome Trace...")
    prof.export_chrome_trace(trace_file)
    
    total_io_time = loader.io_time
    total_gb_read = total_bytes_read / (1024 ** 3)
    bandwidth = total_gb_read / total_io_time if total_io_time > 0 else 0

    print("\n" + "="*40)
    print("        PROFILING RESULTS")
    print("="*40)
    print(f"Total I/O Data Read : {total_gb_read:.4f} GB")
    print(f"Total I/O Time      : {total_io_time:.4f} sec")
    print(f"I/O Bandwidth       : {bandwidth:.4f} GB/s")
    print(f"Trace saved to      : {trace_file}")
    print("="*40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--Nt", type=int, default=420)
    parser.add_argument("--k", type=int, default=175)
    parser.add_argument("--r_sq", type=float, default=1.0)
    parser.add_argument("--precision", type=str, choices=["single", "double"], default="single")
    parser.add_argument("--max_evals", type=int, default=50, help="Evals to profile")
    parser.add_argument("--trace_file", type=str, default="io_compute_overlap.json")
    
    args = parser.parse_args()

    run_profiler(
        args.h5_path, args.Nt, args.k, args.max_evals, 
        args.r_sq, args.precision, args.trace_file
    )