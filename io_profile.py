import argparse
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

from cuda_device import resolve_torch_device
from h5_batch_io import build_si_read_plan, load_ii_block, load_si_column_blocks
from pipelined_loader import PipelinedLoader


@dataclass
class TimelineTracer:
    """Thread-safe wall-clock intervals for overlap timeline plots."""

    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    t0: float = field(default_factory=time.perf_counter)

    def now(self) -> float:
        return time.perf_counter()

    def rel_ms(self, t: float) -> float:
        return (t - self.t0) * 1000.0

    def record(self, lane: str, name: str, start: float, end: float, **meta: Any) -> None:
        if end < start:
            return
        event = {
            "lane": lane,
            "name": name,
            "start_ms": self.rel_ms(start),
            "dur_ms": (end - start) * 1000.0,
            **meta,
        }
        with self._lock:
            self.events.append(event)

    def export(self, path: str, **meta: Any) -> None:
        payload = {"format": "pipelined_timeline_v1", "events": self.events, **meta}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def export_breakdown(path: str, candidate: int, io_ms: dict, gpu_ms: dict, **meta: Any) -> None:
    payload = {
        "format": "candidate_breakdown_v1",
        "candidate": candidate,
        "io_ms": io_ms,
        "gpu_ms": gpu_ms,
        **meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _setup_pipeline(h5_path, Nt, k, max_evals, r_sq, precision, device=None):
    if device is None:
        device = resolve_torch_device()
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available!")

    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    compute_dtype = torch.float64 if precision == "double" else torch.float32
    h5_element_size = 8 if precision == "double" else 4

    print(f"Opening HDF5 store at: {h5_path}")
    h5_file = h5py.File(h5_path, mode="r", rdcc_nbytes=0)
    dset_name = "K_matrix" if "K_matrix" in h5_file else list(h5_file.keys())[0]
    h5_dset = h5_file[dset_name]
    Nd = h5_dset.shape[0] // Nt

    h5_dtype = h5_dset.dtype
    torch_h5_dtype = torch.float64 if h5_dtype == np.float64 else torch.float32
    needs_cast = torch_h5_dtype != compute_dtype

    current_size = k * Nt
    L_S = (torch.eye(current_size, dtype=compute_dtype) * math.sqrt(r_sq)).to(device)

    pinned_Si = [
        torch.empty((current_size, Nt), dtype=torch_h5_dtype).pin_memory() for _ in range(2)
    ]
    pinned_ii = [
        torch.empty((Nt, Nt), dtype=torch_h5_dtype).pin_memory() for _ in range(2)
    ]

    K_Si_gpu_raw = [
        torch.empty((current_size, Nt), dtype=torch_h5_dtype, device=device) for _ in range(2)
    ]
    K_ii_gpu_raw = [
        torch.empty((Nt, Nt), dtype=torch_h5_dtype, device=device) for _ in range(2)
    ]

    K_Si_gpu_math = (
        [torch.empty((current_size, Nt), dtype=compute_dtype, device=device) for _ in range(2)]
        if needs_cast
        else K_Si_gpu_raw
    )
    K_ii_gpu_math = (
        [torch.empty((Nt, Nt), dtype=compute_dtype, device=device) for _ in range(2)]
        if needs_cast
        else K_ii_gpu_raw
    )

    dummy_info = [torch.empty((), dtype=torch.int32, device=device) for _ in range(2)]
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    gpu_done_events = [torch.cuda.Event() for _ in range(2)]
    for event, stream in zip(gpu_done_events, streams):
        event.record(stream)

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
    bytes_per_candidate = (k * Nt * Nt + Nt * Nt) * h5_element_size

    return {
        "device": device,
        "h5_file": h5_file,
        "h5_dset": h5_dset,
        "compute_dtype": compute_dtype,
        "needs_cast": needs_cast,
        "L_S": L_S,
        "pinned_Si": pinned_Si,
        "pinned_ii": pinned_ii,
        "K_Si_gpu_raw": K_Si_gpu_raw,
        "K_ii_gpu_raw": K_ii_gpu_raw,
        "K_Si_gpu_math": K_Si_gpu_math,
        "K_ii_gpu_math": K_ii_gpu_math,
        "dummy_info": dummy_info,
        "streams": streams,
        "gpu_done_events": gpu_done_events,
        "candidate_sequence": candidate_sequence,
        "si_read_plan": si_read_plan,
        "si_slab_np": si_slab_np,
        "Nt": Nt,
        "bytes_per_candidate": bytes_per_candidate,
        "r_sq": r_sq,
    }


def _dispatch_gpu(ctx, curr_b, segment_events=None):
    needs_cast = ctx["needs_cast"]
    r_sq = ctx["r_sq"]
    stream = ctx["streams"][curr_b]

    def _mark():
        if segment_events is None:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        segment_events.append(event)

    with torch.cuda.stream(stream):
        _mark()
        ctx["K_Si_gpu_raw"][curr_b].copy_(ctx["pinned_Si"][curr_b], non_blocking=True)
        _mark()
        ctx["K_ii_gpu_raw"][curr_b].copy_(ctx["pinned_ii"][curr_b], non_blocking=True)
        _mark()

        if needs_cast:
            ctx["K_Si_gpu_math"][curr_b].copy_(
                ctx["K_Si_gpu_raw"][curr_b], non_blocking=True
            )
            _mark()
            ctx["K_ii_gpu_math"][curr_b].copy_(
                ctx["K_ii_gpu_raw"][curr_b], non_blocking=True
            )
            _mark()

        ctx["K_Si_gpu_math"][curr_b].mul_(r_sq)
        _mark()
        torch.linalg.solve_triangular(
            ctx["L_S"],
            ctx["K_Si_gpu_math"][curr_b],
            upper=False,
            out=ctx["K_Si_gpu_math"][curr_b],
        )
        _mark()

        ctx["K_ii_gpu_math"][curr_b].mul_(r_sq).diagonal().add_(1.0)
        _mark()
        ctx["K_ii_gpu_math"][curr_b].addmm_(
            ctx["K_Si_gpu_math"][curr_b].T,
            ctx["K_Si_gpu_math"][curr_b],
            alpha=-1.0,
            beta=1.0,
        )
        _mark()

        torch.linalg.cholesky_ex(
            ctx["K_ii_gpu_math"][curr_b],
            check_errors=False,
            out=(ctx["K_ii_gpu_math"][curr_b], ctx["dummy_info"][curr_b]),
        )
        _mark()

        ctx["gpu_done_events"][curr_b].record(stream)


def _gpu_segment_names(needs_cast):
    names = ["h2d_si", "h2d_ii"]
    if needs_cast:
        names += ["cast_si", "cast_ii"]
    names += ["scale_si", "trsm", "scale_ii_diag", "schur_gemm", "chol"]
    return names


def _record_gpu_segments(tracer, wall_start, markers, names, candidate, buffer):
    if len(markers) < 2 or len(names) != len(markers) - 1:
        return

    offset_ms = 0.0
    for idx, seg_name in enumerate(names):
        seg_ms = markers[idx].elapsed_time(markers[idx + 1])
        if seg_ms <= 0:
            continue
        tracer.record(
            "gpu",
            seg_name,
            wall_start + offset_ms / 1000.0,
            wall_start + (offset_ms + seg_ms) / 1000.0,
            candidate=candidate,
            buffer=buffer,
        )
        offset_ms += seg_ms


def _load_candidate_io(ctx, c_idx, buf_idx, record=None):
    """Load one candidate into pinned buffers; optionally record per-phase times (ms)."""
    col_start = ctx["candidate_sequence"][c_idx] * ctx["Nt"]
    col_end = col_start + ctx["Nt"]
    np_Si_pinned_view = ctx["pinned_Si"][buf_idx].numpy()
    np_ii_pinned_view = ctx["pinned_ii"][buf_idx].numpy()

    t_si0 = time.perf_counter()
    load_si_column_blocks(
        ctx["h5_dset"],
        col_start,
        col_end,
        ctx["Nt"],
        np_Si_pinned_view,
        ctx["si_read_plan"],
        slab_view=ctx["si_slab_np"],
    )
    t_si1 = time.perf_counter()

    t_ii0 = time.perf_counter()
    load_ii_block(ctx["h5_dset"], col_start, col_end, np_ii_pinned_view)
    t_ii1 = time.perf_counter()

    if record is not None:
        record["hdf5_read_si"] = (t_si1 - t_si0) * 1000.0
        record["hdf5_read_ii"] = (t_ii1 - t_ii0) * 1000.0

    return t_ii1 - t_si0


def _print_io_summary(total_bytes_read, loader, trace_file, extra_lines=None):
    total_io_time = loader.io_time
    total_gb_read = total_bytes_read / (1024**3)
    bandwidth = total_gb_read / total_io_time if total_io_time > 0 else 0.0

    print("\n" + "=" * 40)
    print("        PROFILING RESULTS")
    print("=" * 40)
    print(f"Total I/O Data Read : {total_gb_read:.4f} GB")
    print(f"Total I/O Time      : {total_io_time:.4f} sec")
    print(f"I/O Bandwidth       : {bandwidth:.4f} GB/s")
    if extra_lines:
        for line in extra_lines:
            print(line)
    print(f"Trace saved to      : {trace_file}")
    print("=" * 40 + "\n")


def _record_gpu_ms(markers, names):
    gpu_ms = {}
    if len(markers) < 2 or len(names) != len(markers) - 1:
        return gpu_ms
    for idx, seg_name in enumerate(names):
        seg_ms = markers[idx].elapsed_time(markers[idx + 1])
        if seg_ms > 0:
            gpu_ms[seg_name] = seg_ms
    return gpu_ms


def run_breakdown(
    h5_path,
    Nt,
    k,
    max_evals,
    r_sq,
    precision,
    breakdown_file,
    breakdown_candidate,
):
    if breakdown_candidate < 0 or breakdown_candidate >= max_evals:
        raise ValueError(
            f"--breakdown_candidate must be in [0, {max_evals - 1}], "
            f"got {breakdown_candidate}"
        )

    ctx = _setup_pipeline(h5_path, Nt, k, max_evals, r_sq, precision)
    io_ms: dict[str, float] = {}
    gpu_ms: dict[str, float] = {}
    segment_names = _gpu_segment_names(ctx["needs_cast"])
    total_bytes_read = 0
    target = breakdown_candidate

    class _BreakdownTracer:
        def record(self, lane, name, start, end, **meta):
            if meta.get("candidate") != target or lane != "io":
                return
            io_ms[name] = io_ms.get(name, 0.0) + (end - start) * 1000.0

    breakdown_tracer = _BreakdownTracer()

    def load_candidate_to_buffer(c_idx, buf_idx):
        nonlocal total_bytes_read
        record = io_ms if c_idx == target else None
        elapsed = _load_candidate_io(ctx, c_idx, buf_idx, record=record)
        total_bytes_read += ctx["bytes_per_candidate"]
        return elapsed

    loader = PipelinedLoader(
        load_candidate_to_buffer,
        cuda_device=ctx["device"],
        tracer=breakdown_tracer,
    )

    if target == 0:
        t0 = time.perf_counter()
        ctx["gpu_done_events"][0].synchronize()
        io_ms["gpu_buffer_wait"] = (time.perf_counter() - t0) * 1000.0
        loader.load_sync(0, 0)
    else:
        loader.load_sync(0, 0)
    torch.cuda.synchronize()

    print(
        f"\nMeasuring I/O and GPU breakdown for candidate {target} "
        f"(after {target} warmup evals)..."
    )

    for c in tqdm(range(target + 1)):
        curr_b = c % 2
        next_b = (c + 1) % 2

        if c > 0:
            loader.wait()

        if c + 1 < max_evals and c < target:
            loader.start_async(
                c + 1,
                next_b,
                before_load=ctx["gpu_done_events"][next_b].synchronize,
            )

        if c == target:
            markers = []
            with torch.cuda.stream(ctx["streams"][curr_b]):
                _dispatch_gpu(ctx, curr_b, markers)
            torch.cuda.synchronize()
            gpu_ms.update(_record_gpu_ms(markers, segment_names))
        else:
            with torch.cuda.stream(ctx["streams"][curr_b]):
                _dispatch_gpu(ctx, curr_b)

    loader.wait()
    torch.cuda.synchronize()
    ctx["h5_file"].close()

    export_breakdown(
        breakdown_file,
        candidate=target,
        io_ms=io_ms,
        gpu_ms=gpu_ms,
        k=k,
        Nt=Nt,
    )

    io_total = sum(io_ms.values())
    gpu_total = sum(gpu_ms.values())
    print("\n" + "=" * 40)
    print("     CANDIDATE BREAKDOWN")
    print("=" * 40)
    print(f"Candidate           : {target}")
    print(f"I/O total           : {io_total:.2f} ms")
    for name in IO_BREAKDOWN_ORDER:
        if name in io_ms:
            print(f"  {name:18s}: {io_ms[name]:6.2f} ms ({100 * io_ms[name] / io_total:5.1f}%)")
    print(f"GPU compute total   : {gpu_total:.2f} ms")
    for name in segment_names:
        if name in gpu_ms:
            print(f"  {name:18s}: {gpu_ms[name]:6.2f} ms ({100 * gpu_ms[name] / gpu_total:5.1f}%)")
    print(f"Breakdown saved to  : {breakdown_file}")
    print("=" * 40 + "\n")


IO_BREAKDOWN_ORDER = ["gpu_buffer_wait", "hdf5_read_si", "hdf5_read_ii", "hdf5_read"]


def run_timeline(
    h5_path,
    Nt,
    k,
    max_evals,
    r_sq,
    precision,
    trace_file,
    start_candidate,
    num_plot_candidates,
    timeline_record_from=0,
):
    if timeline_record_from < 0 or timeline_record_from >= max_evals:
        raise ValueError(
            f"--timeline_record_from must be in [0, {max_evals - 1}], "
            f"got {timeline_record_from}"
        )

    timeline_capture_from = max(0, timeline_record_from - 1)
    timeline_begin_iter = (
        max(0, timeline_capture_from - 1) if timeline_record_from > 0 else 0
    )

    ctx = _setup_pipeline(h5_path, Nt, k, max_evals, r_sq, precision)

    tracer = TimelineTracer()
    record_state = {"enabled": timeline_capture_from == 0, "tracer": tracer}
    total_bytes_read = 0
    gpu_wall_starts = [0.0, 0.0]
    io_prefetch_start: dict[int, float] = {}
    loader = None

    def _begin_recording(c):
        record_state["enabled"] = True
        record_state["tracer"] = TimelineTracer()
        loader.tracer = record_state["tracer"]
        print(
            f"Recording timeline from candidate {timeline_capture_from} "
            f"(plot window starts at {timeline_record_from}; loop index {c})."
        )

    def _sync_and_record_gpu(buf_idx, candidate):
        ctx["gpu_done_events"][buf_idx].synchronize()
        if not record_state["enabled"] or candidate < timeline_capture_from:
            return
        t_end = record_state["tracer"].now()
        record_state["tracer"].record(
            "gpu",
            "compute",
            gpu_wall_starts[buf_idx],
            t_end,
            candidate=candidate,
            buffer=buf_idx,
        )

    def load_candidate_to_buffer(c_idx, buf_idx):
        nonlocal total_bytes_read
        elapsed = _load_candidate_io(ctx, c_idx, buf_idx)
        total_bytes_read += ctx["bytes_per_candidate"]
        return elapsed

    loader = PipelinedLoader(
        load_candidate_to_buffer,
        cuda_device=ctx["device"],
        tracer=None,
    )
    if record_state["enabled"]:
        io_prefetch_start[0] = tracer.now()
    loader.load_sync(0, 0)
    if record_state["enabled"]:
        record_state["tracer"].record(
            "io",
            "hdf5_read",
            io_prefetch_start[0],
            record_state["tracer"].now(),
            candidate=0,
            buffer=0,
        )
    torch.cuda.synchronize()

    if timeline_record_from > 0:
        print(
            f"\nRunning {timeline_begin_iter} warmup evals, then capturing overlap "
            f"timeline from candidate {timeline_capture_from} through "
            f"candidate {max_evals - 1} (plot window starts at "
            f"{timeline_record_from})..."
        )
    else:
        print(f"\nCapturing overlap timeline for {max_evals} candidates...")

    for c in tqdm(range(max_evals)):
        if c == timeline_begin_iter and timeline_record_from > 0:
            _begin_recording(c)

        curr_b = c % 2
        next_b = (c + 1) % 2

        if c > 0:
            loader.wait()
            if (
                record_state["enabled"]
                and c >= timeline_capture_from
                and c in io_prefetch_start
            ):
                record_state["tracer"].record(
                    "io",
                    "hdf5_read",
                    io_prefetch_start[c],
                    record_state["tracer"].now(),
                    candidate=c,
                    buffer=curr_b,
                )

        gpu_wall_starts[curr_b] = (
            record_state["tracer"].now() if record_state["enabled"] else time.perf_counter()
        )
        with torch.cuda.stream(ctx["streams"][curr_b]):
            _dispatch_gpu(ctx, curr_b)

        if c + 1 < max_evals:
            gpu_cand = c - 1
            sync_buf = next_b

            def _before_load(candidate=gpu_cand, buf=sync_buf):
                if candidate >= 0:
                    _sync_and_record_gpu(buf, candidate)
                else:
                    ctx["gpu_done_events"][buf].synchronize()

            if record_state["enabled"]:
                io_prefetch_start[c + 1] = record_state["tracer"].now()
            loader.start_async(c + 1, next_b, before_load=_before_load)

    loader.wait()
    torch.cuda.synchronize()

    if max_evals >= 2:
        _sync_and_record_gpu((max_evals - 2) % 2, max_evals - 2)
    if max_evals >= 1:
        _sync_and_record_gpu((max_evals - 1) % 2, max_evals - 1)

    ctx["h5_file"].close()
    plot_start = start_candidate if start_candidate > 0 else timeline_record_from
    record_state["tracer"].export(
        trace_file,
        start_candidate=plot_start,
        num_candidates=num_plot_candidates,
        focus_candidate=plot_start,
        timeline_record_from=timeline_record_from,
        timeline_capture_from=timeline_capture_from,
        max_evals=max_evals,
        k=k,
        Nt=Nt,
    )
    _print_io_summary(
        total_bytes_read,
        loader,
        trace_file,
        extra_lines=[
            f"Events recorded     : {len(record_state['tracer'].events)}",
            f"Plot window starts  : candidate {timeline_record_from}",
            f"Capture starts at   : candidate {timeline_capture_from}",
        ],
    )


def run_profiler(h5_path, Nt, k, max_evals, r_sq, precision, trace_file):
    # Single-GPU profiling script: avoid multi-device probe overhead.
    ctx = _setup_pipeline(
        h5_path, Nt, k, max_evals, r_sq, precision, device=torch.device("cuda:0")
    )
    total_bytes_read = 0

    def load_candidate_to_buffer(c_idx, buf_idx):
        nonlocal total_bytes_read

        t0 = time.perf_counter()
        col_start = ctx["candidate_sequence"][c_idx] * ctx["Nt"]
        col_end = col_start + ctx["Nt"]

        with record_function("POSIX_Lustre_Read"):
            np_Si_pinned_view = ctx["pinned_Si"][buf_idx].numpy()
            np_ii_pinned_view = ctx["pinned_ii"][buf_idx].numpy()
            load_si_column_blocks(
                ctx["h5_dset"],
                col_start,
                col_end,
                ctx["Nt"],
                np_Si_pinned_view,
                ctx["si_read_plan"],
                slab_view=ctx["si_slab_np"],
            )
            load_ii_block(ctx["h5_dset"], col_start, col_end, np_ii_pinned_view)

        total_bytes_read += ctx["bytes_per_candidate"]
        return time.perf_counter() - t0

    loader = PipelinedLoader(load_candidate_to_buffer, cuda_device=ctx["device"])
    loader.load_sync(0, 0)
    torch.cuda.synchronize()

    print(f"\nStarting PyTorch profiler loop for {max_evals} candidates...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for c in tqdm(range(max_evals)):
            curr_b = c % 2
            next_b = (c + 1) % 2

            with record_function(f"Candidate_Eval_{c}"):
                if c > 0:
                    with record_function("Wait_For_IO"):
                        loader.wait()

                if c + 1 < max_evals:

                    def _sync_next_buffer(event=ctx["gpu_done_events"][next_b]):
                        with record_function("IO_Wait_For_GPU"):
                            event.synchronize()

                    loader.start_async(c + 1, next_b, before_load=_sync_next_buffer)

                with record_function("GPU_Stream_Dispatch"):
                    with torch.cuda.stream(ctx["streams"][curr_b]):
                        with record_function("H2D_Transfer"):
                            ctx["K_Si_gpu_raw"][curr_b].copy_(
                                ctx["pinned_Si"][curr_b], non_blocking=True
                            )
                            ctx["K_ii_gpu_raw"][curr_b].copy_(
                                ctx["pinned_ii"][curr_b], non_blocking=True
                            )

                        with record_function("Math_Operations"):
                            if ctx["needs_cast"]:
                                ctx["K_Si_gpu_math"][curr_b].copy_(
                                    ctx["K_Si_gpu_raw"][curr_b], non_blocking=True
                                )
                                ctx["K_ii_gpu_math"][curr_b].copy_(
                                    ctx["K_ii_gpu_raw"][curr_b], non_blocking=True
                                )

                            ctx["K_Si_gpu_math"][curr_b].mul_(ctx["r_sq"])
                            torch.linalg.solve_triangular(
                                ctx["L_S"],
                                ctx["K_Si_gpu_math"][curr_b],
                                upper=False,
                                out=ctx["K_Si_gpu_math"][curr_b],
                            )

                            ctx["K_ii_gpu_math"][curr_b].mul_(ctx["r_sq"]).diagonal().add_(1.0)
                            ctx["K_ii_gpu_math"][curr_b].addmm_(
                                ctx["K_Si_gpu_math"][curr_b].T,
                                ctx["K_Si_gpu_math"][curr_b],
                                alpha=-1.0,
                                beta=1.0,
                            )

                            torch.linalg.cholesky_ex(
                                ctx["K_ii_gpu_math"][curr_b],
                                check_errors=False,
                                out=(
                                    ctx["K_ii_gpu_math"][curr_b],
                                    ctx["dummy_info"][curr_b],
                                ),
                            )

                        ctx["gpu_done_events"][curr_b].record(ctx["streams"][curr_b])

        loader.wait()
        torch.cuda.synchronize()

    ctx["h5_file"].close()
    print("\nExporting Chrome trace...")
    prof.export_chrome_trace(trace_file)
    _print_io_summary(total_bytes_read, loader, trace_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile pipelined I/O overlap (PyTorch trace or wall-clock timeline)"
    )
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--Nt", type=int, default=420)
    parser.add_argument("--k", type=int, default=175)
    parser.add_argument("--r_sq", type=float, default=1.0)
    parser.add_argument("--precision", type=str, choices=["single", "double"], default="single")
    parser.add_argument("--max_evals", type=int, default=50, help="Evals to profile")
    parser.add_argument("--trace_file", type=str, default=None)
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Capture wall-clock overlap timeline (recommended for overlap plots)",
    )
    parser.add_argument(
        "--start_candidate",
        type=int,
        default=0,
        help="First candidate shown by plot_io_trace.py (timeline mode only)",
    )
    parser.add_argument(
        "--num_plot_candidates",
        type=int,
        default=3,
        help="Number of candidate evals shown in overlap plot (timeline mode only)",
    )
    parser.add_argument(
        "--timeline_record_from",
        type=int,
        default=0,
        help=(
            "Plot-window start candidate. Timeline events are captured from "
            "one candidate earlier (x-1) so the prior GPU stream tail is included."
        ),
    )
    parser.add_argument(
        "--focus_candidate",
        type=int,
        default=None,
        help="Deprecated alias for --start_candidate",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Profile I/O and GPU sub-ops for one candidate (separate from timeline)",
    )
    parser.add_argument(
        "--breakdown_candidate",
        type=int,
        default=0,
        help="Candidate index for --breakdown (default: 0)",
    )
    parser.add_argument(
        "--breakdown_file",
        type=str,
        default="io_candidate_breakdown.json",
        help="JSON output for --breakdown",
    )

    args = parser.parse_args()
    start_candidate = args.start_candidate
    if args.focus_candidate is not None:
        start_candidate = args.focus_candidate
    trace_file = args.trace_file or (
        "io_overlap_timeline.json" if args.timeline else "io_compute_overlap.json"
    )

    if args.breakdown:
        run_breakdown(
            args.h5_path,
            args.Nt,
            args.k,
            max(args.max_evals, args.breakdown_candidate + 2),
            args.r_sq,
            args.precision,
            args.breakdown_file,
            args.breakdown_candidate,
        )
    elif args.timeline:
        run_timeline(
            args.h5_path,
            args.Nt,
            args.k,
            args.max_evals,
            args.r_sq,
            args.precision,
            trace_file,
            start_candidate,
            args.num_plot_candidates,
            args.timeline_record_from,
        )
    else:
        run_profiler(
            args.h5_path,
            args.Nt,
            args.k,
            args.max_evals,
            args.r_sq,
            args.precision,
            trace_file,
        )
