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

from h5_batch_io import build_si_read_plan, load_ii_block, load_si_column_blocks
from io_profile import TimelineTracer
from pipelined_loader import PipelinedLoader


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_benchmark(
    filename,
    h5_path,
    total_candidates,
    Nt,
    k,
    runs,
    max_evals,
    r_sq,
    seed_s,
    seed_candidates,
    timeline_file=None,
    timeline_record_from=0,
    timeline_start_candidate=0,
    timeline_num_candidates=10,
):
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
    si_read_plan = build_si_read_plan(mock_S_indices, Nt)
    max_si_rows = max(
        (entry["h5_row_end"] - entry["h5_row_start"] for entry in si_read_plan),
        default=Nt,
    )
    si_slab = torch.empty((max_si_rows, Nt), dtype=torch_h5_dtype).pin_memory()
    si_slab_np = si_slab.numpy()
    if rank == 0:
        print(
            f"Batched HDF5 reads: {len(si_read_plan)} hyperslabs for {k} sensor rows "
            f"(was {k} per candidate)"
        )
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

        return MPI.Wtime() - t0

    if rank == 0 and timeline_file:
        if timeline_record_from < 0 or timeline_record_from >= actual_evals:
            print(
                f"Error: --timeline_record_from must be in [0, {actual_evals - 1}], "
                f"got {timeline_record_from}"
            )
            comm.Abort()
        timeline_capture_from = max(0, timeline_record_from - 1)
        print(
            f"Rank-0 overlap timeline: recording on final run "
            f"(plot window candidates {timeline_record_from}..{actual_evals - 1}, "
            f"capture from {timeline_capture_from}) -> {timeline_file}"
        )
    else:
        timeline_capture_from = max(0, timeline_record_from - 1)

    timeline_begin_iter = (
        max(0, timeline_capture_from - 1) if timeline_record_from > 0 else 0
    )

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

        timeline_active = rank == 0 and timeline_file is not None and r == runs - 1
        record_state = None
        gpu_wall_starts = [0.0, 0.0]
        io_prefetch_start: dict[int, float] = {}

        if timeline_active:
            record_state = {
                "enabled": timeline_capture_from == 0,
                "tracer": TimelineTracer(),
            }

        def _sync_and_record_gpu(buf_idx, candidate):
            gpu_done_events[buf_idx].synchronize()
            if (
                not record_state
                or not record_state["enabled"]
                or candidate < timeline_capture_from
            ):
                return
            tracer = record_state["tracer"]
            tracer.record(
                "gpu",
                "compute",
                gpu_wall_starts[buf_idx],
                tracer.now(),
                candidate=candidate,
                buffer=buf_idx,
            )

        def _begin_recording(c):
            nonlocal record_state
            record_state["enabled"] = True
            record_state["tracer"] = TimelineTracer()
            loader.tracer = record_state["tracer"]
            if rank == 0:
                print(
                    f"Recording timeline from candidate {timeline_capture_from} "
                    f"(plot window starts at {timeline_record_from}; loop index {c})."
                )

        loader = PipelinedLoader(
            load_candidate_to_buffer,
            cuda_device=device,
            tracer=record_state["tracer"] if record_state and record_state["enabled"] else None,
        )

        if actual_evals > 0:
            if timeline_active and record_state and record_state["enabled"]:
                io_prefetch_start[0] = record_state["tracer"].now()
            loader.load_sync(0, buf_idx=0)
            if timeline_active and record_state and record_state["enabled"]:
                tracer = record_state["tracer"]
                tracer.record(
                    "io",
                    "hdf5_read",
                    io_prefetch_start[0],
                    tracer.now(),
                    candidate=0,
                    buffer=0,
                )
            torch.cuda.synchronize()

        for c in tqdm(range(actual_evals), disable=rank != 0):
            curr_b = c % 2
            next_b = (c + 1) % 2

            nvtx.range_push(f"Candidate_Loop_{c}")

            if timeline_active and c == timeline_begin_iter and timeline_record_from > 0:
                _begin_recording(c)

            if c > 0:
                loader.wait()
                if (
                    timeline_active
                    and record_state
                    and record_state["enabled"]
                    and c >= timeline_capture_from
                    and c in io_prefetch_start
                ):
                    tracer = record_state["tracer"]
                    tracer.record(
                        "io",
                        "hdf5_read",
                        io_prefetch_start[c],
                        tracer.now(),
                        candidate=c,
                        buffer=curr_b,
                    )

            if c >= 2:
                done_b = (c - 2) % 2
                local_compute_time += (
                    start_events[done_b].elapsed_time(end_events[done_b]) / 1000.0
                )

            if timeline_active and record_state and record_state["enabled"]:
                gpu_wall_starts[curr_b] = record_state["tracer"].now()

            # Prefetch on the I/O thread; GPU-buffer sync also runs there.
            if c + 1 < actual_evals:
                gpu_cand = c - 1
                sync_buf = next_b

                def _before_load(candidate=gpu_cand, buf=sync_buf):
                    if timeline_active and record_state and record_state["enabled"]:
                        if candidate >= 0:
                            _sync_and_record_gpu(buf, candidate)
                        else:
                            gpu_done_events[buf].synchronize()
                    else:
                        gpu_done_events[buf].synchronize()

                if timeline_active and record_state and record_state["enabled"]:
                    io_prefetch_start[c + 1] = record_state["tracer"].now()
                loader.start_async(c + 1, next_b, before_load=_before_load)

            # ------------------------------------------------
            # 1. ASYNC GPU PIPELINE (ISOLATED STREAM)
            # ------------------------------------------------
            with torch.cuda.stream(streams[curr_b]):
                nvtx.range_push("GPU_Dispatch")

                K_Si_gpu_raw[curr_b].copy_(pinned_Si[curr_b], non_blocking=True)
                K_ii_gpu_raw[curr_b].copy_(pinned_ii[curr_b], non_blocking=True)

                if needs_cast:
                    K_Si_gpu_math[curr_b].copy_(K_Si_gpu_raw[curr_b], non_blocking=True)
                    K_ii_gpu_math[curr_b].copy_(K_ii_gpu_raw[curr_b], non_blocking=True)

                start_events[curr_b].record(streams[curr_b])

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

            nvtx.range_pop()

        loader.wait()
        torch.cuda.synchronize()

        if timeline_active and record_state and record_state["enabled"]:
            if actual_evals >= 2:
                _sync_and_record_gpu((actual_evals - 2) % 2, actual_evals - 2)
            if actual_evals >= 1:
                _sync_and_record_gpu((actual_evals - 1) % 2, actual_evals - 1)

            plot_start = (
                timeline_start_candidate
                if timeline_start_candidate > 0
                else timeline_record_from
            )
            num_plot = (
                timeline_num_candidates
                if timeline_num_candidates > 0
                else actual_evals - plot_start
            )
            record_state["tracer"].export(
                timeline_file,
                start_candidate=plot_start,
                num_candidates=num_plot,
                focus_candidate=plot_start,
                timeline_record_from=timeline_record_from,
                timeline_capture_from=timeline_capture_from,
                max_evals=actual_evals,
                k=k,
                Nt=Nt,
                mpi_ranks=size,
                rank0_only=True,
            )
            print(
                f"Wrote rank-0 overlap timeline ({len(record_state['tracer'].events)} events) "
                f"to {timeline_file}"
            )

        if actual_evals >= 1:
            last_b = (actual_evals - 1) % 2
            local_compute_time += (
                start_events[last_b].elapsed_time(end_events[last_b]) / 1000.0
            )
        if actual_evals >= 2:
            prev_b = (actual_evals - 2) % 2
            local_compute_time += (
                start_events[prev_b].elapsed_time(end_events[prev_b]) / 1000.0
            )

        local_best_score = max(
            local_best_score_gpu[0].item(), local_best_score_gpu[1].item()
        )

        local_io_time = loader.io_time
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
    parser.add_argument(
        "--timeline",
        type=str,
        default=None,
        help="Write rank-0 pipelined overlap timeline JSON (recorded on the last run only).",
    )
    parser.add_argument(
        "--timeline_record_from",
        type=int,
        default=50,
        help=(
            "Plot-window start candidate. Timeline events are captured from "
            "one candidate earlier (x-1) so the prior GPU stream tail is included."
        ),
    )
    parser.add_argument(
        "--timeline_start_candidate",
        type=int,
        default=0,
        help="First candidate shown by plot_io_trace.py (0 uses --timeline_record_from).",
    )
    parser.add_argument(
        "--timeline_num_candidates",
        type=int,
        default=10,
        help="Number of candidates in the overlap plot window.",
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
        timeline_file=args.timeline,
        timeline_record_from=args.timeline_record_from,
        timeline_start_candidate=args.timeline_start_candidate,
        timeline_num_candidates=args.timeline_num_candidates,
    )
