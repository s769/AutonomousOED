import json
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

def plot_trace(trace_file, output_file="overlap_trace.pdf"):
    if not os.path.exists(trace_file):
        print(f"Error: Trace file '{trace_file}' not found.")
        return

    print(f"Loading trace data from {trace_file}...")
    with open(trace_file, 'r') as f:
        trace_data = json.load(f)

    events = trace_data.get('traceEvents', [])

    # 1. Find the time bounds for a single iteration (Candidate_Eval_0)
    # We use Candidate 0 because it contains the math for config 0, 
    # and the overlapped speculative I/O fetch for config 1.
    start_ts = None
    end_ts = None
    for e in events:
        if e.get("name") == "Candidate_Eval_0" and e.get("ph") == "X":
            start_ts = e.get("ts")
            end_ts = start_ts + e.get("dur")
            break

    if start_ts is None:
        print("Error: Could not find 'Candidate_Eval_0' in trace. Ensure the profiler ran successfully.")
        return

    # Add a tiny 2% padding to the time window for visual breathing room
    window_dur = end_ts - start_ts
    plot_start_ts = start_ts - (window_dur * 0.02)
    plot_end_ts = end_ts + (window_dur * 0.05)

    # 2. Extract strictly Compute and I/O events within this window
    io_blocks = []
    compute_blocks = []
    has_lustre_read = False
    has_wait_for_io = False

    for e in events:
        if e.get("ph") != "X":
            continue  # Only look at Complete (X) events

        name = e.get("name", "")
        ts = e.get("ts")
        dur = e.get("dur")

        if ts is None or dur is None:
            continue

        # Check if the event intersects our plot window
        if ts + dur < plot_start_ts or ts > plot_end_ts:
            continue

        # Normalize time to milliseconds relative to the start of the plot window
        start_ms = (ts - plot_start_ts) / 1000.0
        dur_ms = dur / 1000.0

        if "POSIX_Lustre_Read" in name:
            has_lustre_read = True
            io_blocks.append((start_ms, dur_ms))
        elif name in ("Wait_For_IO", "IO_Wait_For_GPU", "Wait_For_Next_Buffer"):
            has_wait_for_io = True
            io_blocks.append((start_ms, dur_ms))
        elif "Math_Operations" in name or "H2D_Transfer" in name:
            # We group the transfer and math together as the GPU Compute block
            compute_blocks.append((start_ms, dur_ms))

    # 3. Plotting
    print(f"Found {len(io_blocks)} I/O blocks and {len(compute_blocks)} Compute blocks in the window.")
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=300)
    fig.patch.set_facecolor('#0F0E0D')
    ax.set_facecolor('#0F0E0D')

    if has_lustre_read:
        io_label = "CPU File I/O"
        legend_io = "HDF5 POSIX Read"
    elif has_wait_for_io:
        io_label = "I/O Wait (prefetch)"
        legend_io = "Main-thread I/O stall"
    else:
        io_label = "CPU File I/O"
        legend_io = "HDF5 POSIX Read"
    y_labels = ["GPU Compute", io_label]
    y_positions = [1, 0]

    # Plot CPU I/O (Silver/White for contrast)
    for start, dur in io_blocks:
        ax.barh(0, dur, left=start, height=0.4, color='#DDDDDD', edgecolor='white', linewidth=0.5)

    # Plot GPU Compute (AMD Accent Green)
    for start, dur in compute_blocks:
        ax.barh(1, dur, left=start, height=0.4, color='#329874', edgecolor='white', linewidth=0.5)

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, color='white', fontweight='bold')
    ax.set_xlabel("Time (milliseconds)", color='white', fontweight='bold')
    ax.set_title("Single-Iteration Pipelined Overlap Trace", color='white', fontweight='bold')

    ax.tick_params(colors='white')
    for spine in ['bottom', 'left']: ax.spines[spine].set_color('white')
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)

    # Add a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#DDDDDD', edgecolor='white', label=legend_io),
        Patch(facecolor='#329874', edgecolor='white', label='Schur Update (cuSOLVER)')
    ]
    legend = ax.legend(handles=legend_elements, loc="upper right", frameon=True, 
                       facecolor='#0F0E0D', edgecolor='white')
    for text in legend.get_texts(): 
        text.set_color("white")

    ax.grid(True, axis='x', alpha=0.15, color='white', linestyle='--')
    ax.set_ylim(-0.5, 1.7)

    plt.tight_layout()
    plt.savefig(output_file, facecolor=fig.get_facecolor())
    print(f"Trace plot successfully saved to '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PyTorch JSON Trace as a Timeline")
    parser.add_argument("--trace_file", type=str, default="io_compute_overlap.json", 
                        help="Path to the JSON trace generated by PyTorch profiler")
    parser.add_argument("--output", type=str, default="overlap_trace.pdf", 
                        help="Output PDF filename")
    args = parser.parse_args()

    plot_trace(args.trace_file, args.output)