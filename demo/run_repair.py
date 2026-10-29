#!/usr/bin/env python3
"""Run a D-opt *repair* after an exclusion and write demo/repairs/<id>.json.

Keeps feasible reference sensors, drops excluded ones, then greedily fills
back to the original budget with --exclude_file.

Example (interactive allocation already held):

  srun -u --jobid=$JOB -N 4 --ntasks-per-node=4 -c 32 --gpus-per-task=1 --gpu-bind=none \\
    python demo/run_repair.py --scenario branch_south --jobid $JOB
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="branch_south | branch_north")
    ap.add_argument(
        "--data",
        default="/pscratch/sd/s/srvenkat/oed_demo/data",
    )
    ap.add_argument(
        "--reference",
        default="/global/homes/s/srvenkat/testing/certificates/oed175_rebuilt_select_single_n4.txt",
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
        help="Load K, rebuild the retained factor, then wait for a go file before greedy fill.",
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    sc_path = os.path.join(args.data, "scenarios", f"{args.scenario}.json")
    with open(sc_path) as f:
        sc = json.load(f)
    excl_path = os.path.join(args.data, "scenarios", f"{args.scenario}_exclude.txt")
    out_ckpt = os.path.join(args.data, "repairs", f"{args.scenario}_select.txt")
    out_json = os.path.join(args.data, "repairs", f"{args.scenario}.json")
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)

    ref = np.loadtxt(args.reference)
    ref_ids = ref[:, 0].astype(int)
    excl = set(sc["exclude"])
    retained = [int(s) for s in ref_ids if int(s) not in excl]
    n_lost = len(ref_ids) - len(retained)

    # Seed restart file with retained sensors (logdet placeholder 0; select rebuilds).
    seed = os.path.join(args.data, "repairs", f"{args.scenario}_seed.txt")
    np.savetxt(
        seed,
        np.column_stack([retained, np.zeros(len(retained))]),
        fmt="%d %.18e",
    )

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
        "--exclude_file",
        excl_path,
        "--restart_from",
        seed,
        "--checkpoint_file",
        out_ckpt,
    ]
    go_file = os.path.join(args.data, "repairs", f"{args.scenario}_go")
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

    print("Retained", len(retained), "lost", n_lost)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return

    env = os.environ.copy()
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    env["PYTHONUNBUFFERED"] = "1"
    # Optional extra site-packages (legacy pip h5py). Do not prepend by
    # default — it shadows a conda-built parallel h5py.
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
    new_order = ckpt[:, 0].astype(int).tolist()
    logdets = ckpt[:, 1].astype(float)
    retained_set = set(retained)
    replacement_ids = [s for s in new_order if s not in retained_set]
    n_new = len(replacement_ids)
    lost_ids = [int(s) for s in ref_ids if int(s) in excl]

    orig_logdet = None
    orig_f = None
    pres_path = os.path.join(args.data, "presentation.json")
    if os.path.isfile(pres_path):
        pres = json.load(open(pres_path))
        orig_logdet = pres.get("reference", {}).get("logdet_K")
        orig_f = pres.get("reference", {}).get("information_gain_f")
    if orig_logdet is None:
        ref_tab = np.loadtxt(args.reference)
        orig_logdet = float(ref_tab[-1, 1])
    if orig_f is None:
        orig_f = 0.5 * float(orig_logdet)

    n_ret = len(retained)
    g_rem = float(logdets[n_ret - 1]) if n_ret else 0.0
    g_rep = float(logdets[-1])
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
    f_rem = 0.5 * g_rem
    f_rep = 0.5 * g_rep
    ig = {
        "definition": "f(S)=0.5*logdet(K_S); K=I+r_sq*C, R=I so logdet(R_S)=0. Natural units (nats).",
        "original": orig_f,
        "remainder": f_rem,
        "repaired": f_rep,
        "relative_to_original": {
            "remainder": f_rem / orig_f if orig_f else None,
            "repaired": f_rep / orig_f if orig_f else None,
        },
        "recovered_by_replacements": f_rep - f_rem,
        "monotone_ok": bool(f_rep + 1e-3 >= f_rem),
    }

    prev = {}
    if os.path.isfile(out_json):
        try:
            prev = json.load(open(out_json))
        except Exception:
            prev = {}
    same = prev.get("order") == new_order
    frame_png = prev.get("frame_png") if same else None
    uq_ok = bool(same and prev.get("uq_available") and frame_png)

    payload = {
        "scenario": args.scenario,
        "order": new_order,
        "n_lost": n_lost,
        "n_new": n_new,
        "n_retained": n_ret,
        "lost_ids": lost_ids,
        "retained_ids": retained,
        "replacement_ids": replacement_ids,
        "selection_file": out_ckpt,
        "frame_png": frame_png,
        "uq_available": uq_ok,
        "uq_mean_std": prev.get("uq_mean_std") if uq_ok else None,
        "precomputed": False,
        "live": True,
        "wait_for_go": bool(args.wait_for_go),
        "wall_seconds": round(float(selection_seconds), 1),
        "wall_seconds_total": round(wall, 1),
        "selection_seconds": round(float(selection_seconds), 1),
        "jobid": args.jobid,
        "hardware": "%s GPU nodes (job %s)" % (args.nodes, args.jobid or "local"),
        "policy": "Keep feasible reference sensors; greedy D-opt fill back to 175 under --exclude_file.",
        "information_gain_f": ig,
        "logdet_K": {
            "original": orig_logdet,
            "remainder": g_rem,
            "repaired": g_rep,
            "note": "Production unnormalized logdet(K_S) from select_sensors.py (single precision).",
        },
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print("Wrote", out_json, "wall_seconds", round(wall, 1), flush=True)


if __name__ == "__main__":
    main()
