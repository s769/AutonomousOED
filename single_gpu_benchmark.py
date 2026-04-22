import torch
import gc
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.transforms as transforms


def cleanup():
    """Forces the Python garbage collector and empties the GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_csv_rows(filename):
    """Load benchmark rows from CSV if available."""
    if not os.path.isfile(filename):
        return {}

    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)

    existing_data = {}
    for row in data:
        existing_data[int(row[0])] = list(row[1:])
    return existing_data


def run_benchmark(filename, Nt, max_budget, step, runs):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available! This benchmark requires an active GPU."
        )

    device = torch.device("cuda:0")
    dtype = torch.float32

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # --- SMART RESUMPTION & PATCHING LOGIC ---
    existing_data = {}
    file_exists = os.path.isfile(filename)

    if file_exists:
        try:
            existing_data = _load_csv_rows(filename)
            print(f"Loaded existing data from {filename}. Will patch missing values.")
        except Exception as e:
            print(f"Could not read existing file: {e}. Starting from scratch.")

    print(
        f"{'k':<4} | {'Mem: N-OOP':<10} | {'N-IP':<8} | {'S-OOP':<8} | {'S-IP':<8} || {'Time: N-OOP':<11} | {'N-IP':<8} | {'S-OOP':<8} | {'S-IP':<8}"
    )
    print("-" * 105)

    results = []

    # Independent OOM Latches
    oom_n_oop = False
    oom_n_ip = False
    oom_s_oop = False
    oom_s_ip = False

    with torch.no_grad():
        for k in range(10, max_budget + 1, step):
            current_size = k * Nt

            # Initialize defaults
            vals = {
                "m_no": float("nan"),
                "m_ni": float("nan"),
                "m_so": float("nan"),
                "m_si": float("nan"),
                "t_no": float("nan"),
                "t_ni": float("nan"),
                "t_so": float("nan"),
                "t_si": float("nan"),
            }
            needs = {
                "n_oop": not oom_n_oop,
                "n_ip": not oom_n_ip,
                "s_oop": not oom_s_oop,
                "s_ip": not oom_s_ip,
            }

            # Override with existing data if present
            if k in existing_data:
                e = existing_data[k]
                vals["m_no"], vals["m_ni"], vals["m_so"], vals["m_si"] = (
                    e[0],
                    e[1],
                    e[2],
                    e[3],
                )
                vals["t_no"], vals["t_ni"], vals["t_so"], vals["t_si"] = (
                    e[4],
                    e[5],
                    e[6],
                    e[7],
                )

                if not np.isnan(vals["m_no"]) and not np.isnan(vals["t_no"]):
                    needs["n_oop"] = False
                if not np.isnan(vals["m_ni"]) and not np.isnan(vals["t_ni"]):
                    needs["n_ip"] = False
                if not np.isnan(vals["m_so"]) and not np.isnan(vals["t_so"]):
                    needs["s_oop"] = False
                if not np.isnan(vals["m_si"]) and not np.isnan(vals["t_si"]):
                    needs["s_ip"] = False

            if not any(needs.values()):
                print(
                    f"{k:<4} | {vals['m_no']:<10.3f} | {vals['m_ni']:<8.3f} | {vals['m_so']:<8.3f} | {vals['m_si']:<8.3f} || {vals['t_no']:<11.3f} | {vals['t_ni']:<8.3f} | {vals['t_so']:<8.3f} | {vals['t_si']:<8.3f} (Skip)"
                )
                results.append(
                    [
                        k,
                        vals["m_no"],
                        vals["m_ni"],
                        vals["m_so"],
                        vals["m_si"],
                        vals["t_no"],
                        vals["t_ni"],
                        vals["t_so"],
                        vals["t_si"],
                    ]
                )
                continue

            cleanup()

            # Identity Trick for guaranteed positive-definiteness
            L_S_cpu = torch.eye(current_size, dtype=dtype)
            K_Si_cpu = torch.zeros(current_size, Nt, dtype=dtype)
            K_ii_cpu = torch.eye(Nt, dtype=dtype)

            # ==========================================
            # 1. NAIVE OUT-OF-PLACE (N-OOP)
            # ==========================================
            if needs["n_oop"]:
                cleanup()
                torch.cuda.reset_peak_memory_stats()
                try:
                    L_S, K_Si, K_ii = (
                        L_S_cpu.to(device),
                        K_Si_cpu.to(device),
                        K_ii_cpu.to(device),
                    )

                    # Pre-allocate all distinct buffers for strict OOP
                    dummy_L = torch.empty(
                        (current_size + Nt, current_size + Nt),
                        dtype=dtype,
                        device=device,
                    )
                    dummy_info = torch.empty((), dtype=torch.int32, device=device)
                    K_S = torch.empty(
                        (current_size, current_size), dtype=dtype, device=device
                    )
                    K_aug = torch.empty(
                        (current_size + Nt, current_size + Nt),
                        dtype=dtype,
                        device=device,
                    )

                    # Peak Memory Run (Math populates the pre-allocated buffers)
                    torch.mm(L_S, L_S.T, out=K_S)
                    K_aug[:current_size, :current_size] = K_S
                    K_aug[:current_size, current_size:] = K_Si
                    K_aug[current_size:, :current_size] = K_Si.T
                    K_aug[current_size:, current_size:] = K_ii
                    torch.linalg.cholesky_ex(K_aug, out=(dummy_L, dummy_info))

                    vals["m_no"] = torch.cuda.max_memory_allocated() / (1024**3)

                    # Timing Loop (Zero allocations inside)
                    t_accum = 0.0
                    for _ in range(runs):
                        start_event.record()
                        torch.mm(L_S, L_S.T, out=K_S)
                        K_aug[:current_size, :current_size] = K_S
                        K_aug[:current_size, current_size:] = K_Si
                        K_aug[current_size:, :current_size] = K_Si.T
                        K_aug[current_size:, current_size:] = K_ii
                        torch.linalg.cholesky_ex(
                            K_aug, check_errors=False, out=(dummy_L, dummy_info)
                        )
                        end_event.record()
                        torch.cuda.synchronize()
                        t_accum += start_event.elapsed_time(end_event) / 1000.0
                    vals["t_no"] = t_accum / runs

                except torch.cuda.OutOfMemoryError:
                    oom_n_oop = True
                L_S = K_Si = K_ii = K_S = K_aug = dummy_L = dummy_info = None

            # ==========================================
            # 2. NAIVE IN-PLACE (N-IP)
            # ==========================================
            if needs["n_ip"]:
                cleanup()
                torch.cuda.reset_peak_memory_stats()
                try:
                    L_S, K_Si, K_ii = (
                        L_S_cpu.to(device),
                        K_Si_cpu.to(device),
                        K_ii_cpu.to(device),
                    )

                    # Pre-allocate ONLY the final target buffer
                    dummy_info = torch.empty((), dtype=torch.int32, device=device)
                    K_aug = torch.empty(
                        (current_size + Nt, current_size + Nt),
                        dtype=dtype,
                        device=device,
                    )

                    # Peak Memory Run (In-place operations, skipping intermediate K_S tensor)
                    torch.mm(L_S, L_S.T, out=K_aug[:current_size, :current_size])
                    K_aug[:current_size, current_size:] = K_Si
                    K_aug[current_size:, :current_size] = K_Si.T
                    K_aug[current_size:, current_size:] = K_ii
                    # True Aliased In-Place Cholesky (Input is Output)
                    torch.linalg.cholesky_ex(K_aug, out=(K_aug, dummy_info))

                    vals["m_ni"] = torch.cuda.max_memory_allocated() / (1024**3)

                    # Timing Loop (Zero allocations inside).
                    # FIX: start_event must be recorded BEFORE assembly, not after.
                    # N-OOP times assembly + Cholesky; N-IP must do the same so the
                    # comparison is apples-to-apples. The in-place Cholesky overwrites
                    # K_aug, so the assembly must run every iteration anyway.
                    t_accum = 0.0
                    for _ in range(runs):
                        start_event.record()  # <-- moved before assembly (was after)
                        torch.mm(L_S, L_S.T, out=K_aug[:current_size, :current_size])
                        K_aug[:current_size, current_size:] = K_Si
                        K_aug[current_size:, :current_size] = K_Si.T
                        K_aug[current_size:, current_size:] = K_ii
                        torch.linalg.cholesky_ex(
                            K_aug, check_errors=False, out=(K_aug, dummy_info)
                        )
                        end_event.record()
                        torch.cuda.synchronize()
                        t_accum += start_event.elapsed_time(end_event) / 1000.0
                    vals["t_ni"] = t_accum / runs

                except torch.cuda.OutOfMemoryError:
                    oom_n_ip = True
                L_S = K_Si = K_ii = K_aug = dummy_info = None

            # ==========================================
            # 3. SCHUR OUT-OF-PLACE (S-OOP)
            # ==========================================
            if needs["s_oop"]:
                cleanup()
                torch.cuda.reset_peak_memory_stats()
                try:
                    L_S, K_Si, K_ii = (
                        L_S_cpu.to(device),
                        K_Si_cpu.to(device),
                        K_ii_cpu.to(device),
                    )

                    # Pre-allocate all distinct buffers
                    dummy_L = torch.empty((Nt, Nt), dtype=dtype, device=device)
                    dummy_info = torch.empty((), dtype=torch.int32, device=device)
                    Y = torch.empty_like(K_Si)
                    M = torch.empty_like(K_ii)

                    # Peak Memory Run
                    torch.linalg.solve_triangular(L_S, K_Si, upper=False, out=Y)
                    torch.addmm(K_ii, Y.T, Y, alpha=-1.0, beta=1.0, out=M)
                    torch.linalg.cholesky_ex(M, out=(dummy_L, dummy_info))

                    vals["m_so"] = torch.cuda.max_memory_allocated() / (1024**3)

                    # Timing Loop (Zero allocations inside)
                    t_accum = 0.0
                    for _ in range(runs):
                        start_event.record()
                        torch.linalg.solve_triangular(L_S, K_Si, upper=False, out=Y)
                        torch.addmm(K_ii, Y.T, Y, alpha=-1.0, beta=1.0, out=M)
                        torch.linalg.cholesky_ex(
                            M, check_errors=False, out=(dummy_L, dummy_info)
                        )
                        end_event.record()
                        torch.cuda.synchronize()
                        t_accum += start_event.elapsed_time(end_event) / 1000.0
                    vals["t_so"] = t_accum / runs

                except torch.cuda.OutOfMemoryError:
                    oom_s_oop = True
                L_S = K_Si = K_ii = Y = M = dummy_L = dummy_info = None

            # ==========================================
            # 4. SCHUR STRICT IN-PLACE (S-IP)
            # ==========================================
            if needs["s_ip"]:
                cleanup()
                torch.cuda.reset_peak_memory_stats()
                try:
                    L_S, K_Si, K_ii = (
                        L_S_cpu.to(device),
                        K_Si_cpu.to(device),
                        K_ii_cpu.to(device),
                    )

                    dummy_info = torch.empty((), dtype=torch.int32, device=device)
                    # Clone inputs since they will be irreversibly destroyed
                    K_Si_tmp = K_Si.clone()
                    K_ii_tmp = K_ii.clone()

                    # Peak Memory Run
                    # 1. Overwrite K_Si_tmp directly
                    torch.linalg.solve_triangular(
                        L_S, K_Si_tmp, upper=False, out=K_Si_tmp
                    )
                    # 2. Overwrite K_ii_tmp directly.
                    # FIX: use addmm_ (the explicit in-place method) instead of
                    # addmm(..., out=K_ii_tmp) where K_ii_tmp is also the `input`
                    # argument. Passing the same tensor as both `input` and `out` in
                    # addmm is not formally supported by cuBLAS and can produce
                    # incorrect results. addmm_ is the correct API for in-place
                    # beta*self + alpha*mat1@mat2 with self as both accumulator and output.
                    K_ii_tmp.addmm_(K_Si_tmp.T, K_Si_tmp, alpha=-1.0, beta=1.0)
                    # 3. True Aliased In-Place Cholesky
                    torch.linalg.cholesky_ex(K_ii_tmp, out=(K_ii_tmp, dummy_info))

                    vals["m_si"] = torch.cuda.max_memory_allocated() / (1024**3)

                    # Timing Loop (Zero allocations inside)
                    t_accum = 0.0
                    for _ in range(runs):
                        # Reset buffers OUTSIDE the stopwatch
                        K_Si_tmp.copy_(K_Si)
                        K_ii_tmp.copy_(K_ii)

                        start_event.record()
                        torch.linalg.solve_triangular(
                            L_S, K_Si_tmp, upper=False, out=K_Si_tmp
                        )
                        K_ii_tmp.addmm_(K_Si_tmp.T, K_Si_tmp, alpha=-1.0, beta=1.0)
                        torch.linalg.cholesky_ex(
                            K_ii_tmp, check_errors=False, out=(K_ii_tmp, dummy_info)
                        )
                        end_event.record()

                        torch.cuda.synchronize()
                        t_accum += start_event.elapsed_time(end_event) / 1000.0
                    vals["t_si"] = t_accum / runs

                except torch.cuda.OutOfMemoryError:
                    oom_s_ip = True
                L_S = K_Si = K_ii = K_Si_tmp = K_ii_tmp = dummy_info = None

            L_S_cpu = K_Si_cpu = K_ii_cpu = None

            print(
                f"{k:<4} | {vals['m_no']:<10.3f} | {vals['m_ni']:<8.3f} | {vals['m_so']:<8.3f} | {vals['m_si']:<8.3f} || {vals['t_no']:<11.3f} | {vals['t_ni']:<8.3f} | {vals['t_so']:<8.3f} | {vals['t_si']:<8.3f}"
            )

            row = [
                k,
                vals["m_no"],
                vals["m_ni"],
                vals["m_so"],
                vals["m_si"],
                vals["t_no"],
                vals["t_ni"],
                vals["t_so"],
                vals["t_si"],
            ]
            results.append(row)

            # Save atomically
            header = "k,mem_N_OOP,mem_N_IP,mem_S_OOP,mem_S_IP,time_N_OOP,time_N_IP,time_S_OOP,time_S_IP"
            np.savetxt(
                filename, np.array(results), delimiter=",", header=header, comments=""
            )


def plot_results(filename, arch):
    sns.set_context("paper", font_scale=1.3)
    sns.set_palette("colorblind")
    sns.set_style("whitegrid")
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)
    k_vals = data[:, 0]

    # Extract Memory
    m_no, m_ni, m_so, m_si = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    # Extract Time
    t_no, t_ni, t_so, t_si = data[:, 5], data[:, 6], data[:, 7], data[:, 8]

    # Define styling dictionary to keep plots consistent
    styles = {
        "N_OOP": {
            "marker": "o",
            "linestyle": "-",
            "label": "Naive (Out-of-Place)",
        },
        "N_IP": {
            "marker": "v",
            "linestyle": "--",
            "label": "Naive (In-Place)",
        },
        "S_OOP": {
            "marker": "^",
            "linestyle": ":",
            "label": "Schur Update (Out-of-Place)",
        },
        "S_IP": {
            "marker": "s",
            "linestyle": "-",
            "label": "Schur Update (In-Place)",
        },
    }

    # ---------------------------
    # PLOT 1: MEMORY
    # ---------------------------
    fig1, ax1 = plt.subplots(figsize=(6, 4.5), dpi=300)

    plot_data_mem = [
        (m_no, "N_OOP"),
        (m_ni, "N_IP"),
        (m_so, "S_OOP"),
        (m_si, "S_IP"),
    ]

    for arr, key in plot_data_mem:
        mask = np.isfinite(arr)
        ax1.plot(k_vals[mask], arr[mask], linewidth=2.5, markersize=6, **styles[key])

    gpu_details = {
        "mi250x": {"name": "AMD MI250X", "vram": 64.0, "label": "MI250X VRAM LIMIT"},
        "a100": {"name": "NVIDIA A100", "vram": 80.0, "label": "A100 VRAM LIMIT"},
        "gh200": {"name": "NVIDIA GH200", "vram": 96.0, "label": "GH200 VRAM LIMIT"},
    }

    max_vram = max(details["vram"] for details in gpu_details.values())
    ax1.set_ylim(0, max_vram * 1.15)

    trans = transforms.blended_transform_factory(ax1.transAxes, ax1.transData)

    for arc in gpu_details:
        ax1.axhline(
            y=gpu_details[arc]["vram"],
            color="#555555",
            linestyle="--",
            alpha=0.8,
        )
        ax1.text(
            0.02,
            gpu_details[arc]["vram"] + 1,
            gpu_details[arc]["label"],
            transform=trans,
            fontweight="bold",
            fontsize=10,
            color="black",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        )

    ax1.set_xlabel(r"Number of Selected Sensors", fontweight="bold")
    ax1.set_ylabel(r"Peak GPU VRAM Allocated (GB)", fontweight="bold")
    ax1.tick_params(axis="both", which="major")
    ax1.legend(loc="lower right", frameon=True, shadow=True)

    plt.tight_layout()
    mem_plot_file = filename.replace(".csv", "_memory.pdf")
    fig1.savefig(mem_plot_file, format="pdf", bbox_inches="tight")
    print(f"Memory plot saved to {mem_plot_file}")

    # ---------------------------
    # PLOT 2: TIME WITH INSET
    # ---------------------------
    fig2, ax2 = plt.subplots(figsize=(6, 4.5), dpi=300)

    plot_data_time = [(t_no, "N_OOP"), (t_ni, "N_IP"), (t_so, "S_OOP"), (t_si, "S_IP")]

    for arr, key in plot_data_time:
        mask = np.isfinite(arr)
        ax2.plot(k_vals[mask], arr[mask], **styles[key])

    ax2.set_xlabel(r"Number of Selected Sensors ($k$)", fontweight="bold")
    ax2.set_ylabel("Compute Time per Candidate (s)", fontweight="bold")
    ax2.tick_params(axis="both", which="major")
    ax2.legend(loc="upper left", frameon=True, shadow=True)

    # --- CREATE THE ZOOMED INSET ---
    valid_so = np.isfinite(t_so)
    valid_si = np.isfinite(t_si)

    if np.any(valid_so) and np.any(valid_si):
        axins = ax2.inset_axes([0.55, 0.25, 0.4, 0.4])

        axins.plot(k_vals[valid_so], t_so[valid_so], **styles["S_OOP"])
        axins.plot(k_vals[valid_si], t_si[valid_si], **styles["S_IP"])

        max_k_valid = max(k_vals[valid_so].max(), k_vals[valid_si].max())
        zoom_start_k = max_k_valid * 0.75
        axins.set_xlim(zoom_start_k, max_k_valid)

        zoom_mask_so = valid_so & (k_vals >= zoom_start_k)
        zoom_mask_si = valid_si & (k_vals >= zoom_start_k)

        if np.any(zoom_mask_so) and np.any(zoom_mask_si):
            min_y = min(t_so[zoom_mask_so].min(), t_si[zoom_mask_si].min())
            max_y = max(t_so[zoom_mask_so].max(), t_si[zoom_mask_si].max())
            y_margin = (max_y - min_y) * 0.15 if max_y > min_y else max_y * 0.15 + 1e-12
            axins.set_ylim(min_y - y_margin, max_y + y_margin)

        axins.tick_params(labelsize=9)
        axins.grid(True, linestyle=":", alpha=0.6)
        ax2.indicate_inset_zoom(axins, edgecolor="black", alpha=0.3)

    plt.tight_layout()
    time_plot_file = filename.replace(".csv", "_time_inset.pdf")
    fig2.savefig(time_plot_file, format="pdf", bbox_inches="tight")
    print(f"Time plot with inset saved to {time_plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="ablation_results_420_single.csv")
    parser.add_argument("--Nt", type=int, default=420)
    parser.add_argument("--max_budget", type=int, default=600)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--arch", type=str, default="a100")
    args = parser.parse_args()

    if args.plot_only:
        plot_results(args.file, args.arch)
    else:
        run_benchmark(
            args.file,
            Nt=args.Nt,
            max_budget=args.max_budget,
            step=args.step,
            runs=args.runs,
        )
