import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

# --- Paper styling ---
sns.set_context("paper", font_scale=1.3)
sns.set_style("whitegrid")
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

LANE_ORDER = ["io", "gpu0", "gpu1"]
LANE_LABELS = {
    "io": "I/O",
    "gpu0": "GPU stream 0",
    "gpu1": "GPU stream 1",
}
IO_COLOR = "#B8B8B8"
GPU_COLORS = {
    "gpu0": "#2E8B57",
    "gpu1": "#3DA56A",
}
IO_EVENT_NAMES = {"gpu_buffer_wait", "hdf5_read"}
BAR_HEIGHT = 0.5


def _detect_format(trace_data):
    if trace_data.get("format") == "pipelined_timeline_v1":
        return "manual"
    if "traceEvents" in trace_data:
        return "pytorch"
    raise ValueError("Unrecognized trace format.")


def _events_for_plot(events, start_candidate, num_candidates=None):
    """GPU: candidates [x-1, x+3]; I/O: candidates [x, x+4]."""
    del num_candidates  # window is fixed relative to start_candidate
    gpu_min = start_candidate - 1 if start_candidate > 0 else start_candidate
    gpu_max = start_candidate + 3
    io_min = start_candidate
    io_max = start_candidate + 4

    selected = []
    for e in events:
        c = e.get("candidate")
        if c is None:
            continue
        if e["lane"] == "io" and io_min <= c <= io_max and e["name"] in IO_EVENT_NAMES:
            selected.append(e)
        elif e["lane"] == "gpu" and e["name"] == "compute" and gpu_min <= c <= gpu_max:
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


def _window_from_events(events, pad_frac=0.03):
    if not events:
        return 0.0, 1.0

    start = min(e["start_ms"] for e in events)
    end = max(e["start_ms"] + e["dur_ms"] for e in events)
    span = max(end - start, 1.0)
    return start - pad_frac * span, end + pad_frac * span


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


def _lane_color(lane):
    if lane == "io":
        return IO_COLOR
    return GPU_COLORS.get(lane, GPU_COLORS["gpu0"])


def plot_manual_timeline(
    trace_data,
    output_file,
    start_candidate=None,
    num_candidates=None,
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

    plot_events = _merge_io_events(filtered) + [
        {**e, "lane": _gpu_plot_lane(e)} for e in filtered if e["lane"] == "gpu"
    ]

    plot_start, plot_end = _window_from_events(plot_events)
    visible = [
        e
        for e in plot_events
        if e["start_ms"] + e["dur_ms"] >= plot_start and e["start_ms"] <= plot_end
    ]

    fig, ax = plt.subplots(figsize=(10, 2.2), dpi=300)

    y_positions = _lane_y_positions(LANE_ORDER)

    for event in visible:
        lane = event["lane"]
        y = y_positions[lane]
        left = event["start_ms"] - plot_start
        width = event["dur_ms"]
        color = _lane_color(lane)

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
        if candidate is not None and width > 12:
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

    ax.set_yticks([y_positions[lane] for lane in LANE_ORDER])
    ax.set_yticklabels([LANE_LABELS[lane] for lane in LANE_ORDER], fontweight="bold")
    ax.set_xlabel("Time (ms)", fontweight="bold")
    ax.tick_params(colors="black")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.grid(True, axis="x", alpha=0.3, color="gray", linestyle="--")
    ax.set_ylim(0, BAR_HEIGHT * len(LANE_ORDER))

    plt.tight_layout()
    plt.savefig(output_file, facecolor="white")
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
            color=GPU_COLOR,
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
):
    if focus_candidate is not None and start_candidate is None:
        start_candidate = focus_candidate

    if not os.path.exists(trace_file):
        print(f"Error: Trace file '{trace_file}' not found.")
        return

    print(f"Loading trace data from {trace_file}...")
    with open(trace_file, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    trace_format = _detect_format(trace_data)
    if trace_format == "manual":
        plot_manual_timeline(
            trace_data,
            output_file,
            start_candidate=start_candidate,
            num_candidates=num_candidates,
        )
    else:
        plot_pytorch_trace(trace_data, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pipelined I/O overlap timeline")
    parser.add_argument(
        "--trace_file",
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
    args = parser.parse_args()

    plot_trace(
        args.trace_file,
        args.output,
        start_candidate=args.start_candidate,
        num_candidates=args.num_candidates,
        focus_candidate=args.focus_candidate,
    )
