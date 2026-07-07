import torch
import gc
import argparse
import numpy as np
import os


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

                    # Timing Loop: time only the Cholesky. K_S is constant across
                    # candidates within a greedy step (formed once, above), and the
                    # OOP factorization writes to a separate buffer, so K_aug is
                    # preserved and re-factored directly each run.
                    t_accum = 0.0
                    for _ in range(runs):
                        start_event.record()
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

                    # Timing Loop: time only the Cholesky. The aliased in-place
                    # factorization overwrites K_aug, so it is rebuilt before each
                    # run OUTSIDE the timed region (the GEMM/assembly is not timed).
                    t_accum = 0.0
                    for _ in range(runs):
                        # Restore K_aug (destroyed by previous in-place factor); untimed.
                        torch.mm(L_S, L_S.T, out=K_aug[:current_size, :current_size])
                        K_aug[:current_size, current_size:] = K_Si
                        K_aug[current_size:, :current_size] = K_Si.T
                        K_aug[current_size:, current_size:] = K_ii
                        torch.cuda.synchronize()

                        start_event.record()
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
                    K_Si_tmp = K_Si.clone()
                    K_ii_tmp = K_ii.clone()

                    # Peak Memory Run
                    torch.linalg.solve_triangular(
                        L_S, K_Si_tmp, upper=False, out=K_Si_tmp
                    )
                    K_ii_tmp.addmm_(K_Si_tmp.T, K_Si_tmp, alpha=-1.0, beta=1.0)
                    torch.linalg.cholesky_ex(K_ii_tmp, out=(K_ii_tmp, dummy_info))

                    vals["m_si"] = torch.cuda.max_memory_allocated() / (1024**3)

                    # Timing Loop (Zero allocations inside)
                    t_accum = 0.0
                    for _ in range(runs):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="ablation_results_420_single.csv")
    parser.add_argument("--Nt", type=int, default=420)
    parser.add_argument("--max_budget", type=int, default=600)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    run_benchmark(
        args.file,
        Nt=args.Nt,
        max_budget=args.max_budget,
        step=args.step,
        runs=args.runs,
    )
