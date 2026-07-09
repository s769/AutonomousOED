import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# --- Paper styling ---
sns.set_context("paper", font_scale=1.3)
sns.set_style("whitegrid")
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

BAR_HEIGHT = 0.5
MAIN_MIN_MS = 1.0
CANDIDATE_LABEL_MIN_MS = 40.0

BASE_LANE_ORDER = ["main", "io", "gpu0", "gpu1"]
LANE_LABELS = {
    "io": "I/O",
    "main": "Main Thread",
    "gpu0": "GPU Stream 0",
    "gpu1": "GPU Stream 1",
}

IO_COLOR = "#B8B8B8"
MAIN_COLORS = {
    "wait_for_io": "#D35400",
    "await_dispatch": "#E67E22",
}
GPU_COLORS = {
    "gpu0": "#2E8B57",
    "gpu1": "#3DA56A",
}

IO_SIMPLE_NAMES = {"gpu_buffer_wait", "hdf5_read"}
GPU_AGGREGATE = "compute"
MAIN_EVENT_NAMES = {"wait_for_io", "await_dispatch"}
MAIN_LABELS = {
    "wait_for_io": "Wait for prefetch",
    "await_dispatch": "Ready → launch",
}

GPU_BREAKDOWN_MERGE = {
    "cast": [
        "h2d_si",
        "h2d_ii",
        "cast_si",
        "cast_ii",
        "scale_si",
        "scale_ii_diag",
        "scale_ii",
        "add_identity",
    ],
    "trsm": ["trsm"],
    "gemm_chol": ["schur_gemm", "chol"],
}
GPU_DISPLAY_ORDER = ["cast", "trsm", "gemm_chol"]
GPU_BREAKDOWN_COLORS = {
    "cast": "#AA3377",
    "trsm": "#004488",
    "gemm_chol": "#C9A227",
    "other": "#8E44AD",
}
ALWAYS_SHOW_BREAKDOWN = frozenset(GPU_DISPLAY_ORDER)
BREAKDOWN_LABELS = {
    "hdf5_read": "HDF5 Read",
    "cast": "Cast + Preprocess",
    "trsm": "TRSM",
    "gemm_chol": "GEMM + Cholesky",
    "other_gpu": "Misc. GPU",
    **MAIN_LABELS,
}
IN_BAR_LABEL_MIN_PCT = 10.0
MIN_BAR_VIS_PCT = 1.2
MAIN_GRID = {"axis": "x", "alpha": 0.3, "color": "gray", "linestyle": "--"}
BREAKDOWN_MAJOR_GRID = {
    "axis": "x",
    "alpha": 0.3,
    "color": "gray",
    "linestyle": "-",
    "linewidth": 1.2,
}


def _detect_format(trace_data):
    fmt = trace_data.get("format")
    if fmt in ("pipelined_timeline_v1", "pipelined_timeline_v2"):
        return "manual"
    if "traceEvents" in trace_data:
        return "pytorch"
    raise ValueError("Unrecognized trace format.")


def _load_breakdown(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != "candidate_breakdown_v1":
        raise ValueError(f"Expected candidate_breakdown_v1, got {data.get('format')!r}")
    return data


def _events_for_plot(events, start_candidate, num_candidates=None):
    """GPU: candidates [x-1, x+3]; I/O and main: candidates [x, x+4]."""
    del num_candidates
    gpu_min = start_candidate - 1 if start_candidate > 0 else start_candidate
    gpu_max = start_candidate + 3
    io_min = start_candidate
    io_max = start_candidate + 4

    selected = []
    for e in events:
        c = e.get("candidate")
        if c is None:
            continue

        if e["lane"] == "io" and io_min <= c <= io_max:
            if e["name"] in IO_SIMPLE_NAMES:
                selected.append(e)
        elif (
            e["lane"] == "main"
            and io_min <= c <= io_max
            and e["name"] in MAIN_EVENT_NAMES
            and e["dur_ms"] >= MAIN_MIN_MS
        ):
            selected.append({**e, "lane": "main"})
        elif (
            e["lane"] == "gpu"
            and gpu_min <= c <= gpu_max
            and e["name"] == GPU_AGGREGATE
        ):
            selected.append(e)
    return selected


def _merge_io_events(events):
    by_candidate: dict[int, dict] = {}
    for e in events:
        if e["lane"] != "io":
            continue
        c = e["candidate"]
        end_ms = e["start_ms"] + e["dur_ms"]
        if c not in by_candidate:
            by_candidate[c] = {"start_ms": e["start_ms"], "end_ms": end_ms, "candidate": c}
        else:
            by_candidate[c]["start_ms"] = min(by_candidate[c]["start_ms"], e["start_ms"])
            by_candidate[c]["end_ms"] = max(by_candidate[c]["end_ms"], end_ms)

    return [
        {
            "lane": "io",
            "name": "hdf5_io",
            "start_ms": span["start_ms"],
            "dur_ms": span["end_ms"] - span["start_ms"],
            "candidate": span["candidate"],
        }
        for span in sorted(by_candidate.values(), key=lambda item: item["candidate"])
    ]


def _main_covers_interval(main_events, candidate, start_ms, end_ms):
    for event in main_events:
        if event.get("candidate") != candidate:
            continue
        ev_start = event["start_ms"]
        ev_end = event["start_ms"] + event["dur_ms"]
        if ev_start <= start_ms + 0.05 and ev_end >= end_ms - 0.05:
            return True
    return False


def _fill_dispatch_gaps(io_bars, all_events, main_events, start_candidate):
    gpu_min = start_candidate - 1 if start_candidate > 0 else start_candidate
    gpu_max = start_candidate + 3
    gpu_by_candidate = {
        e["candidate"]: e
        for e in all_events
        if e["lane"] == "gpu"
        and e["name"] == GPU_AGGREGATE
        and gpu_min <= e["candidate"] <= gpu_max
    }
    filled = list(main_events)

    for io_event in io_bars:
        candidate = io_event["candidate"]
        gpu_event = gpu_by_candidate.get(candidate)
        if gpu_event is None:
            continue

        io_end = io_event["start_ms"] + io_event["dur_ms"]
        gpu_start = gpu_event["start_ms"]
        gap_ms = gpu_start - io_end
        if gap_ms < MAIN_MIN_MS:
            continue
        if _main_covers_interval(filled, candidate, io_end, gpu_start):
            continue

        filled.append(
            {
                "lane": "main",
                "name": "await_dispatch",
                "start_ms": io_end,
                "dur_ms": gap_ms,
                "candidate": candidate,
            }
        )
    return filled


def _window_from_events(events, pad_frac=0.03):
    if not events:
        return 0.0, 1.0

    start = min(e["start_ms"] for e in events)
    end = max(e["start_ms"] + e["dur_ms"] for e in events)
    span = max(end - start, 1.0)
    return start - pad_frac * span, end + pad_frac * span


def _event_start_ms(event):
    return event["start_ms"]


def _event_end_ms(event):
    return event["start_ms"] + event["dur_ms"]


def _plot_window_bounds(plot_events, start_candidate, num_candidates, pad_frac=0.03):
    """Anchor the left edge at GPU start_candidate and I/O start_candidate+1."""
    anchors = []
    for event in plot_events:
        candidate = event.get("candidate")
        if candidate is None:
            continue
        lane = event["lane"]
        if lane.startswith("gpu") and candidate == start_candidate:
            anchors.append(_event_start_ms(event))
        if lane == "io" and candidate == start_candidate + 1:
            anchors.append(_event_start_ms(event))

    plot_start = min(anchors) if anchors else 0.0

    last_candidate = start_candidate + num_candidates - 1
    window_events = [
        e
        for e in plot_events
        if e.get("candidate") is not None
        and start_candidate <= e["candidate"] <= last_candidate + 1
    ]
    if not window_events:
        window_events = list(plot_events)

    plot_end = max(_event_end_ms(e) for e in window_events)
    span = max(plot_end - plot_start, 1.0)
    return plot_start, plot_end + pad_frac * span


def _candidate_label(candidate):
    return f"Candidate {candidate}"


def _gpu_plot_lane(event):
    buf = event.get("buffer")
    if buf is None:
        buf = event.get("candidate", 0) % 2
    return f"gpu{buf}"


def _lane_y_positions(lane_order):
    return {
        lane: (idx + 0.5) * BAR_HEIGHT for idx, lane in enumerate(reversed(lane_order))
    }


def _event_color(event):
    lane = event["lane"]
    name = event.get("name")
    if lane == "main":
        return MAIN_COLORS.get(name, "#E67E22")
    if lane == "io":
        return IO_COLOR
    if lane.startswith("gpu"):
        return GPU_COLORS.get(lane, GPU_COLORS["gpu0"])
    return IO_COLOR


def _build_plot_events(filtered, all_events, start_candidate):
    del all_events, start_candidate
    io_raw = [e for e in filtered if e["lane"] == "io"]
    io_plot = _merge_io_events(io_raw)
    gpu_plot = [
        {**e, "lane": _gpu_plot_lane(e)}
        for e in filtered
        if e["lane"] == "gpu" and e["name"] == GPU_AGGREGATE
    ]
    return io_plot + gpu_plot


def _legend_handles(visible):
    del visible
    return []


def _consolidate_ms(ms_dict, merge_map):
    consolidated = {}
    used = set()
    for key, sources in merge_map.items():
        total = sum(ms_dict.get(source, 0.0) for source in sources)
        if total > 0:
            consolidated[key] = total
        used.update(sources)
    for name, value in ms_dict.items():
        if name not in used:
            consolidated[name] = consolidated.get(name, 0.0) + value
    return consolidated


def _rect_handle(color, label):
    return plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor=color,
        edgecolor="black",
        linewidth=0.6,
        label=label,
    )


def _breakdown_slices(ms_dict, order, colors, min_runtime_pct, other_label):
    total = sum(ms_dict.values())
    if total <= 0:
        return [], [], []

    ordered_names = [name for name in order if name in ms_dict]
    for name in sorted(ms_dict):
        if name not in ordered_names and name != "_other_raw":
            ordered_names.append(name)

    labels, values, facecolors = [], [], []
    other = 100.0 * ms_dict.get("_other_raw", 0.0) / total
    for name in ordered_names:
        pct = 100.0 * ms_dict[name] / total
        if (
            min_runtime_pct > 0
            and pct < min_runtime_pct
            and name not in ALWAYS_SHOW_BREAKDOWN
        ):
            other += pct
            continue
        labels.append(BREAKDOWN_LABELS.get(name, name.replace("_", " ")))
        values.append(pct)
        facecolors.append(colors.get(name, "#CCCCCC"))

    if other > 0:
        labels.append(other_label)
        values.append(other)
        facecolors.append(colors["other"])
    return labels, values, facecolors


def _display_bar_widths(values):
    """Minimum visual width so tiny segments (e.g. GEMM) remain visible."""
    boosted = [max(v, MIN_BAR_VIS_PCT) if v > 0 else 0.0 for v in values]
    total = sum(boosted)
    if total <= 0:
        return list(values)
    return [100.0 * v / total for v in boosted]


def _breakdown_rows(breakdown, min_runtime_pct):
    gpu_ms = _consolidate_ms(breakdown.get("gpu_ms", {}), GPU_BREAKDOWN_MERGE)
    other_gpu = sum(
        gpu_ms.pop(key) for key in list(gpu_ms) if key not in GPU_DISPLAY_ORDER
    )
    if other_gpu > 0:
        gpu_ms["cast"] = gpu_ms.get("cast", 0.0) + other_gpu

    labels, values, facecolors = _breakdown_slices(
        gpu_ms,
        GPU_DISPLAY_ORDER,
        GPU_BREAKDOWN_COLORS,
        min_runtime_pct,
        BREAKDOWN_LABELS["other_gpu"],
    )
    if not values:
        return []
    return [("GPU", labels, values, facecolors)]


def _breakdown_legend_handles(rows):
    handles = []
    seen = set()
    skip = {BREAKDOWN_LABELS["other_gpu"]}
    for _, labels, _, facecolors in rows:
        for label, color in zip(labels, facecolors):
            if label in seen or label in skip:
                continue
            seen.add(label)
            handles.append(_rect_handle(color, label))
    return handles


def _draw_breakdown_panel(ax, breakdown, min_runtime_pct):
    rows = _breakdown_rows(breakdown, min_runtime_pct)
    if not rows:
        ax.set_visible(False)
        return

    candidate = breakdown.get("candidate", "?")
    y_center = 0.25
    bar_h = 0.42
    for _, labels, values, facecolors in rows:
        display_widths = _display_bar_widths(values)
        left = 0.0
        for value, width, color in zip(values, display_widths, facecolors):
            ax.barh(
                y_center,
                width,
                left=left,
                height=bar_h,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            if value >= IN_BAR_LABEL_MIN_PCT:
                ax.text(
                    left + width / 2,
                    y_center,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                    color="white" if value > 20 else "black",
                )
            left += width

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.72)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticks(range(0, 101, 5), minor=True)
    ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=6.5)
    ax.set_xlabel(
        f"GPU Runtime Breakdown (%)\n— {_candidate_label(candidate)}",
        fontsize=7,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, which="major", **BREAKDOWN_MAJOR_GRID)
    ax.grid(True, which="minor", axis="x", alpha=0.15, color="gray", linestyle=":")

    handles = _breakdown_legend_handles(rows)
    if handles:
        ax.legend(
            [h for h in handles],
            [h.get_label() for h in handles],
            loc="upper right",
            ncol=1,
            fontsize=6,
            frameon=True,
            handlelength=1.2,
            labelspacing=0.35,
            borderaxespad=0.4,
        )


def _place_bottom_legend(ax_legend, handles, ncol=None):
    if not handles:
        ax_legend.set_visible(False)
        return

    if ncol is None:
        ncol = len(handles)
    ax_legend.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        bbox_transform=ax_legend.transAxes,
        ncol=ncol,
        fontsize=9,
        frameon=True,
        handlelength=1.6,
        columnspacing=1.2,
    )


def plot_manual_timeline(
    trace_data,
    output_file,
    start_candidate=None,
    num_candidates=None,
    breakdown=None,
    min_runtime_pct=0,
    legend_y=None,
):
    events = trace_data.get("events", [])
    if start_candidate is None:
        start_candidate = trace_data.get(
            "start_candidate",
            trace_data.get(
                "timeline_record_from",
                trace_data.get("focus_candidate", 0),
            ),
        )
    if num_candidates is None:
        num_candidates = trace_data.get("num_candidates", 3)

    filtered = _events_for_plot(events, start_candidate, num_candidates)
    if not filtered:
        raise ValueError(
            f"No events found for start_candidate={start_candidate} "
            f"(GPU {max(0, start_candidate - 1)}–{start_candidate + 3}, "
            f"I/O {start_candidate}–{start_candidate + 4})."
        )

    plot_events = _build_plot_events(filtered, events, start_candidate)
    lane_order = [lane for lane in BASE_LANE_ORDER if lane != "main"]

    plot_start, plot_end = _plot_window_bounds(
        plot_events, start_candidate, num_candidates
    )
    visible = [
        e
        for e in plot_events
        if e["start_ms"] + e["dur_ms"] >= plot_start and e["start_ms"] <= plot_end
    ]

    fig_height = 2.2
    has_breakdown = breakdown is not None and breakdown.get("gpu_ms")
    label_row = 0.16   # spacer for x-axis labels below the plot
    legend_row = 0.16 if legend_y is None else max(0.10, legend_y)

    if has_breakdown:
        fig = plt.figure(figsize=(10, fig_height + 0.42), dpi=300)
        gs = gridspec.GridSpec(
            2,
            2,
            figure=fig,
            height_ratios=[1.0, label_row],
            width_ratios=[2.25, 0.78],
            wspace=0.12,
            hspace=0.22,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_breakdown = fig.add_subplot(gs[0, 1])
        ax_legend = None
    else:
        fig = plt.figure(figsize=(10, fig_height + 0.42), dpi=300)
        gs = gridspec.GridSpec(
            3,
            1,
            figure=fig,
            height_ratios=[1.0, label_row, legend_row],
            hspace=0.22,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_breakdown = None
        ax_legend = fig.add_subplot(gs[2, 0])

    if ax_legend is not None:
        ax_legend.axis("off")

    y_positions = _lane_y_positions(lane_order)

    for event in visible:
        lane = event["lane"]
        y = y_positions[lane]
        bar_start_ms = max(event["start_ms"], plot_start)
        bar_end_ms = min(event["start_ms"] + event["dur_ms"], plot_end)
        if bar_end_ms <= bar_start_ms:
            continue
        left = bar_start_ms - plot_start
        width = bar_end_ms - bar_start_ms
        color = _event_color(event)

        ax.barh(
            y,
            width,
            left=left,
            height=BAR_HEIGHT,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=1,
        )

        candidate = event.get("candidate")
        if (
            candidate is not None
            and lane != "main"
            and candidate >= start_candidate
            and width >= CANDIDATE_LABEL_MIN_MS
        ):
            label = _candidate_label(candidate)
            text_color = "black" if lane == "io" else "white"
            ax.text(
                left + width / 2,
                y,
                label,
                ha="center",
                va="center",
                fontsize=6.5 if lane.startswith("gpu") else 7,
                color=text_color,
                fontweight="bold",
                zorder=3,
                clip_on=False,
            )

    ax.set_yticks([y_positions[lane] for lane in lane_order])
    ax.set_yticklabels([LANE_LABELS[lane] for lane in lane_order], fontweight="bold")
    ax.set_xlabel("Time (ms)", fontweight="bold", labelpad=10)
    ax.tick_params(colors="black", pad=2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.grid(True, **MAIN_GRID)
    ax.set_ylim(0, BAR_HEIGHT * len(lane_order))

    legend_handles = _legend_handles(visible)
    if has_breakdown:
        _draw_breakdown_panel(ax_breakdown, breakdown, min_runtime_pct)
    elif ax_legend is not None and legend_handles:
        _place_bottom_legend(ax_legend, legend_handles)

    fig.subplots_adjust(
        left=0.15 if has_breakdown else 0.13,
        right=0.98,
        top=0.88,
        bottom=0.20,
    )

    fig.savefig(output_file, facecolor="white", pad_inches=0.15)
    root, ext = os.path.splitext(output_file)
    if ext.lower() == ".pdf":
        fig.savefig(f"{root}.png", facecolor="white", dpi=200, pad_inches=0.15)
    plt.close(fig)
    print(f"Timeline plot saved to '{output_file}'")


def plot_pytorch_trace(trace_data, output_file):
    events = trace_data.get("traceEvents", [])

    start_ts = None
    end_ts = None
    for e in events:
        if e.get("name") == "Candidate_Eval_0" and e.get("ph") == "X":
            start_ts = e.get("ts")
            end_ts = start_ts + e.get("dur")
            break

    if start_ts is None:
        raise ValueError(
            "Could not find 'Candidate_Eval_0' in trace. "
            "Use io_profile.py --timeline for background-thread I/O visibility."
        )

    window_dur = end_ts - start_ts
    plot_start_ts = start_ts - (window_dur * 0.02)
    plot_end_ts = end_ts + (window_dur * 0.05)

    io_blocks = []
    compute_blocks = []

    for e in events:
        if e.get("ph") != "X":
            continue

        name = e.get("name", "")
        ts = e.get("ts")
        dur = e.get("dur")
        if ts is None or dur is None:
            continue
        if ts + dur < plot_start_ts or ts > plot_end_ts:
            continue

        start_ms = (ts - plot_start_ts) / 1000.0
        dur_ms = dur / 1000.0

        if "POSIX_Lustre_Read" in name or name in ("IO_Wait_For_GPU",):
            io_blocks.append((start_ms, dur_ms))
        elif "Math_Operations" in name or "H2D_Transfer" in name:
            compute_blocks.append((start_ms, dur_ms))

    fig, ax = plt.subplots(figsize=(8, 1.6), dpi=300)

    for start, dur in io_blocks:
        ax.barh(
            BAR_HEIGHT / 2,
            dur,
            left=start,
            height=BAR_HEIGHT,
            color=IO_COLOR,
            edgecolor="black",
            linewidth=0.6,
        )
    for start, dur in compute_blocks:
        ax.barh(
            BAR_HEIGHT + BAR_HEIGHT / 2,
            dur,
            left=start,
            height=BAR_HEIGHT,
            color=GPU_COLORS["gpu0"],
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_yticks([BAR_HEIGHT / 2, BAR_HEIGHT + BAR_HEIGHT / 2])
    ax.set_yticklabels(["I/O", "GPU compute"], fontweight="bold")
    ax.set_xlabel("Time (ms)", fontweight="bold")
    ax.tick_params(colors="black")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.grid(True, axis="x", alpha=0.3, color="gray", linestyle="--")
    ax.set_ylim(0, BAR_HEIGHT * 2)

    plt.tight_layout()
    plt.savefig(output_file, facecolor="white")
    print(f"Trace plot saved to '{output_file}'")


def plot_trace(
    trace_file,
    output_file="overlap_trace.pdf",
    start_candidate=None,
    num_candidates=None,
    focus_candidate=None,
    breakdown_file=None,
    min_runtime_pct=0,
    legend_y=None,
):
    if focus_candidate is not None and start_candidate is None:
        start_candidate = focus_candidate

    if not os.path.exists(trace_file):
        print(f"Error: Trace file '{trace_file}' not found.")
        return

    print(f"Loading trace data from {trace_file}...")
    with open(trace_file, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    breakdown = None
    if breakdown_file:
        if not os.path.exists(breakdown_file):
            print(f"Warning: breakdown file '{breakdown_file}' not found; skipping inset.")
        else:
            print(f"Loading breakdown inset from {breakdown_file}...")
            breakdown = _load_breakdown(breakdown_file)

    trace_format = _detect_format(trace_data)
    if trace_format == "manual":
        plot_manual_timeline(
            trace_data,
            output_file,
            start_candidate=start_candidate,
            num_candidates=num_candidates,
            breakdown=breakdown,
            min_runtime_pct=min_runtime_pct,
            legend_y=legend_y,
        )
    else:
        plot_pytorch_trace(trace_data, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pipelined I/O overlap timeline")
    parser.add_argument(
        "--trace_file",
        "--trace",
        type=str,
        default="io_overlap_timeline.json",
        help="JSON from io_profile.py --timeline or legacy PyTorch profiler export",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="overlap_trace.pdf",
        help="Output PDF filename",
    )
    parser.add_argument(
        "--start_candidate",
        type=int,
        default=None,
        help="First candidate to show (default: 0)",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=None,
        help="Number of candidate evals to show (default: 3)",
    )
    parser.add_argument(
        "--focus_candidate",
        type=int,
        default=None,
        help="Deprecated alias for --start_candidate",
    )
    parser.add_argument(
        "--breakdown_file",
        type=str,
        default=None,
        help="JSON from io_profile.py --breakdown; GPU % inset on the timeline",
    )
    parser.add_argument(
        "--min_runtime_pct",
        type=float,
        default=0,
        help=(
            "Breakdown panel only: hide sub-ops below this %% of their lane total "
            "(0 shows all; 1 keeps ops >= 1%%)."
        ),
    )
    parser.add_argument(
        "--legend_y",
        type=float,
        default=None,
        help=(
            "Height of the legend row relative to the plot row "
            "(default: 0.16). Decrease (e.g. 0.12) to tighten the gap; "
            "increase if the legend is clipped."
        ),
    )
    args = parser.parse_args()

    plot_trace(
        args.trace_file,
        args.output,
        start_candidate=args.start_candidate,
        num_candidates=args.num_candidates,
        focus_candidate=args.focus_candidate,
        breakdown_file=args.breakdown_file,
        min_runtime_pct=args.min_runtime_pct,
        legend_y=args.legend_y,
    )
