#!/usr/bin/env python3
"""Run a full unconstrained D-opt selection (budget 175) for the live demo.

Unlike run_repair.py this starts from an empty design — no exclusion, no
retained restart prefix.

Example (interactive allocation already held):

  srun -u --jobid=$JOB -N 4 --ntasks-per-node=4 -c 36 --gpus-per-task=1 --gpu-bind=none \\
    python demo/run_full.py --jobid $JOB
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="/pscratch/sd/s/srvenkat/oed_demo/data",
    )
    ap.add_argument(
        "--k",
        default="/pscratch/sd/s/skw/CDS_SHARED/mw-mesh-420s/files/K_prior_4_2d_rebuilt.h5",
    )
    ap.add_argument("--r_sq", type=float, default=1e4)
    ap.add_argument("--budget", type=int, default=175)
    ap.add_argument("--jobid", type=str, default=None)
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument(
        "--nodelist",
        default="",
        help="Optional Slurm -w host list (e.g. 80GB nodes only).",
    )
    ap.add_argument(
        "--ntasks-per-node",
        type=int,
        default=int(os.environ.get("OED_SRUN_NTASKS_PER_NODE", "4")),
    )
    ap.add_argument(
        "--cpus-per-task",
        type=int,
        default=int(os.environ.get("OED_SRUN_CPUS_PER_TASK", "32")),
    )
    ap.add_argument(
        "--wait_for_go",
        action="store_true",
        help="Load K, then wait for a go file before greedy selection.",
    )
    ap.add_argument(
        "--reference",
        default=os.environ.get("OED_REFERENCE", ""),
        help="Unused for full select; accepted so launch wrappers can pass it.",
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    scenario = "full"
    out_ckpt = os.path.join(args.data, "repairs", f"{scenario}_select.txt")
    out_json = os.path.join(args.data, "repairs", f"{scenario}.json")
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    py_cmd = [
        "python",
        "-u",
        os.path.join(repo, "select_sensors.py"),
        args.k,
        str(args.budget),
        "--r_sq",
        str(args.r_sq),
        "--precision",
        "single",
        "--objective",
        "doptimal",
        "--checkpoint_file",
        out_ckpt,
    ]
    go_file = os.path.join(args.data, "repairs", f"{scenario}_go")
    if args.wait_for_go:
        if os.path.isfile(go_file):
            os.remove(go_file)
        py_cmd.extend(["--wait_for_go", go_file])
        print("Will wait for go file", go_file, flush=True)

    cmd = py_cmd
    if args.jobid:
        wrapper = os.path.join(repo, "demo", "select_wrapper.sh")
        cmd = [
            "srun",
            "-u",
            f"--jobid={args.jobid}",
            "-N",
            str(args.nodes),
            "--ntasks-per-node=%d" % args.ntasks_per_node,
            "-c",
            str(args.cpus_per_task),
            "--gpus-per-task=1",
            "--gpu-bind=none",
            "--export=ALL",
        ]
        if args.nodelist:
            cmd.extend(["-w", args.nodelist])
        cmd.append(wrapper)
        cmd.extend(py_cmd[1:])  # drop leading "python"; wrapper execs python -u

    print("Full unconstrained selection, budget", args.budget, flush=True)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return

    env = os.environ.copy()
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    env["PYTHONUNBUFFERED"] = "1"
    parts = [repo]
    deps = env.get("OED_PYDEPS", "").strip()
    if deps:
        parts.insert(0, deps)
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(parts)
    t0 = time.time()
    subprocess.check_call(cmd, cwd=repo, env=env)
    wall = time.time() - t0

    ckpt = np.loadtxt(out_ckpt)
    if ckpt.ndim == 1:
        ckpt = ckpt.reshape(1, -1)
    new_order = ckpt[:, 0].astype(int).tolist()
    logdets = ckpt[:, 1].astype(float)

    orig_logdet = None
    orig_f = None
    pres_path = os.path.join(args.data, "presentation.json")
    if os.path.isfile(pres_path):
        pres = json.load(open(pres_path))
        orig_logdet = pres.get("reference", {}).get("logdet_K")
        orig_f = pres.get("reference", {}).get("information_gain_f")
    if orig_logdet is None:
        orig_logdet = float(logdets[-1]) if len(logdets) else None
    if orig_f is None and orig_logdet is not None:
        orig_f = 0.5 * float(orig_logdet)

    timing = {}
    timing_path = out_ckpt + ".timing.json"
    if os.path.isfile(timing_path):
        try:
            timing = json.load(open(timing_path))
        except Exception:
            timing = {}
    selection_seconds = timing.get("selection_seconds")
    if selection_seconds is None:
        selection_seconds = round(wall, 1)

    g_final = float(logdets[-1]) if len(logdets) else 0.0
    f_final = 0.5 * g_final

    payload = {
        "scenario": scenario,
        "order": new_order,
        "n_selected": len(new_order),
        "budget": args.budget,
        "selection_file": out_ckpt,
        "precomputed": False,
        "live": True,
        "wait_for_go": bool(args.wait_for_go),
        "wall_seconds": round(float(selection_seconds), 1),
        "wall_seconds_total": round(wall, 1),
        "selection_seconds": round(float(selection_seconds), 1),
        "jobid": args.jobid,
        "hardware": "%s GPU nodes (job %s)" % (args.nodes, args.jobid or "local"),
        "policy": "Unconstrained greedy D-opt from empty design to budget.",
        "information_gain_f": {
            "definition": "f(S)=0.5*logdet(K_S); K=I+r_sq*C, R=I so logdet(R_S)=0. Natural units (nats).",
            "selected": f_final,
            "reference": orig_f,
            "relative_to_reference": (f_final / orig_f) if orig_f else None,
        },
        "logdet_K": {
            "selected": g_final,
            "reference": orig_logdet,
            "note": "Production unnormalized logdet(K_S) from select_sensors.py (single precision).",
        },
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print("Wrote", out_json, "wall_seconds", round(wall, 1), flush=True)


if __name__ == "__main__":
    main()
