#!/usr/bin/env python3
"""Certify how close an existing regular D-optimal design is to the global optimum.

Does not modify or re-run the production greedy solver. Follows a supplied
ordered selection and evaluates submodular a-posteriori upper bounds at chosen
checkpoints.

Supports optional MPI: partition remaining-candidate scans across ranks (each
rank keeps a local Cholesky factor), matching select_sensors.py.

Example (cheap: empty + final prefix only):

  python scripts/certify_d_optimality.py cascadia_K_matrix.h5 \\
      --selection cascadia_optimal_results.txt --r_sq 1.0 \\
      --output_prefix certificates/cascadia_k175

Example (16 GPUs):

  srun -n 16 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none \\
    python scripts/certify_d_optimality.py K_prior_4_2d.h5 \\
      --selection oed-200.txt --r_sq 1e4 \\
      --output_prefix certificates/oed200
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from mpi4py import MPI

# Allow running as scripts/certify_d_optimality.py from the repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cuda_device import resolve_local_rank, resolve_torch_device
from d_opt_certificate import (
    certify_selection,
    format_terminal_summary,
    load_selection_file,
    open_k_matrix,
    parse_checkpoints,
    write_checkpoint_csv,
    write_summary_json,
)


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    return [int(p) for p in parts]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "A-posteriori optimality certificates for regular D-optimal "
            "sensor designs (isotropic noise K = r_sq*C + I)."
        )
    )
    p.add_argument("h5_store", type=str, help="Path to 2D-chunked HDF5 C / K_matrix file.")
    p.add_argument(
        "--selection",
        type=str,
        required=True,
        help=(
            "Ordered selection file: production checkpoint "
            "'sensor_id cumulative_logdet' or a single column of sensor IDs."
        ),
    )
    p.add_argument(
        "--r_sq",
        type=float,
        required=True,
        help=(
            "Isotropic noise scaling used when the design was computed. "
            "Required (same meaning as select_sensors.py): K = r_sq*C + I."
        ),
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HDF5 dataset name (default: K_matrix if present, else first key).",
    )
    p.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Target sensor budget k (default: full selection length).",
    )
    p.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help=(
            "Explicit checkpoint prefix lengths, e.g. '0,50,175'. "
            "Default when no stride/all-prefixes: 0 and k."
        ),
    )
    p.add_argument(
        "--checkpoint_stride",
        type=int,
        default=None,
        help=(
            "Also evaluate every stride prefixes in [0, k]. "
            "Scanning all candidates at many prefixes approaches the cost of "
            "re-running greedy analysis."
        ),
    )
    p.add_argument(
        "--all_prefixes",
        action="store_true",
        help="Evaluate every prefix length 0..k (expensive).",
    )
    p.add_argument(
        "--output_prefix",
        type=str,
        default="d_opt_certificate",
        help="Prefix for CSV/JSON outputs (writes <prefix>_checkpoints.csv and <prefix>_summary.json).",
    )
    p.add_argument(
        "--precision",
        type=str,
        choices=["single", "double"],
        default="double",
        help="Certificate arithmetic precision (default: double / FP64).",
    )
    p.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device (default: auto via cuda_device.resolve_torch_device).",
    )
    p.add_argument(
        "--neg_gain_tol",
        type=float,
        default=1e-10,
        help="Clamp marginal gains in [-tol, 0) to 0 and report the adjustment.",
    )
    p.add_argument(
        "--allow_correlated_noise",
        action="store_true",
        help=(
            "Unsupported. The certificate assumes independent isotropic sensor "
            "blocks; this flag is rejected if set."
        ),
    )
    p.add_argument(
        "--goal_oriented",
        action="store_true",
        help="Unsupported configuration; rejected if set.",
    )
    return p


def resolve_device(args, mpi_rank: int) -> torch.device:
    local_rank = resolve_local_rank(mpi_rank)
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return resolve_torch_device(local_rank=local_rank, mpi_rank=mpi_rank)
    return resolve_torch_device(local_rank=local_rank, mpi_rank=mpi_rank)


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.goal_oriented:
        if rank == 0:
            print(
                "Error: goal-oriented D-optimality certificates are not supported. "
                "This tool only certifies regular D-optimality with independent "
                "isotropic sensor-noise blocks.",
                file=sys.stderr,
            )
        return 2
    if args.allow_correlated_noise:
        if rank == 0:
            print(
                "Error: cross-sensor-correlated noise is not supported. "
                "AutonomousOED uses isotropic R = I after K = r_sq*C + I.",
                file=sys.stderr,
            )
        return 2
    if args.r_sq <= 0.0:
        if rank == 0:
            print(f"Error: --r_sq must be positive (got {args.r_sq})", file=sys.stderr)
        return 2

    device = resolve_device(args, mpi_rank=rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = torch.float64 if args.precision == "double" else torch.float32
    torch.set_grad_enabled(False)

    selected = None
    if rank == 0:
        selected, _logged = load_selection_file(args.selection)
    selected = comm.bcast(selected, root=0)

    budget_k = len(selected) if args.budget is None else args.budget
    if budget_k > len(selected):
        if rank == 0:
            print(
                f"Error: --budget {budget_k} exceeds selection length {len(selected)}",
                file=sys.stderr,
            )
        return 2

    explicit_cps = _parse_int_list(args.checkpoints)
    checkpoints = parse_checkpoints(
        selection_len=len(selected),
        budget_k=budget_k,
        checkpoints=explicit_cps,
        checkpoint_stride=args.checkpoint_stride,
        all_prefixes=args.all_prefixes,
    )

    h5_file, h5_dset, dset_name, Nt, Nd = open_k_matrix(args.h5_store, args.dataset)
    try:
        if rank == 0:
            print(
                f"Opened {args.h5_store} [{dset_name}] shape={h5_dset.shape} "
                f"chunks={h5_dset.chunks} Nt={Nt} Nd={Nd}"
            )
            print(
                f"MPI size={size}, Device[rank0]={device}, precision={args.precision}, "
                f"r_sq={args.r_sq}, k={budget_k}, selection_len={len(selected)}, "
                f"checkpoints={checkpoints}"
            )
            if args.all_prefixes or (
                args.checkpoint_stride is not None and args.checkpoint_stride <= 5
            ):
                print(
                    "Note: dense checkpoint schedules re-scan all remaining candidates "
                    "at each prefix and can approach the cost of the original greedy run."
                )

        results, summary = certify_selection(
            h5_dset=h5_dset,
            Nt=Nt,
            Nd=Nd,
            selected=selected,
            budget_k=budget_k,
            checkpoints=checkpoints,
            r_sq=args.r_sq,
            device=device,
            dtype=dtype,
            neg_tol=args.neg_gain_tol,
            dataset_name=dset_name,
            h5_store=os.path.abspath(args.h5_store),
            comm=comm,
            show_progress=(rank == 0),
        )
    finally:
        h5_file.close()

    if rank == 0:
        csv_path = f"{args.output_prefix}_checkpoints.csv"
        json_path = f"{args.output_prefix}_summary.json"
        write_checkpoint_csv(csv_path, results)
        write_summary_json(json_path, summary)
        print(format_terminal_summary(summary))
        print(f"\nWrote {csv_path}")
        print(f"Wrote {json_path}")
    comm.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
