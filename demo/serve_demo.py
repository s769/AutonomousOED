#!/usr/bin/env python3
"""Serve the Cascadia OED demo UI with data rooted on pscratch.

  python demo/serve_demo.py \\
    --data /pscratch/sd/s/srvenkat/oed_demo/data \\
    --port 8765

POST /api/repair  launches a live GPU repair against a held Slurm allocation.
GET  /api/repair/status  and  /api/alloc  report progress.
"""

import argparse
import functools
import json
import os
import re
import subprocess
import threading
import time
from urllib.parse import unquote, urlparse

import http.server
import socketserver

CONDA_ENV = "/global/common/software/m4767/srvenkat/conda/envs/h5torch-mpi-cu13"
K_DEFAULT = "/pscratch/sd/s/skw/CDS_SHARED/mw-mesh-420s/files/K_prior_4_2d_rebuilt.h5"


def _now():
    return time.time()


def find_slurm_job():
    try:
        out = subprocess.check_output(
            [
                "squeue",
                "-u",
                os.environ.get("USER", "srvenkat"),
                "-h",
                "-o",
                "%i %T %D %q",
            ],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", "replace")
    except Exception as exc:
        return None, str(exc)
    running = []
    pending = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        rec = {
            "jobid": parts[0],
            "state": parts[1],
            "nodes": int(parts[2]),
            "qos": parts[3],
        }
        if rec["state"] == "RUNNING":
            running.append(rec)
        elif rec["state"] == "PENDING":
            pending.append(rec)
    if running:
        return running[0], None
    if pending:
        return pending[0], None
    return None, None


class RepairController(object):
    def __init__(self, data_dir, repo_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.repo_dir = os.path.abspath(repo_dir)
        self.lock = threading.Lock()
        self.state = "idle"
        self.mode = "all"
        self.scenario = None
        self.message = ""
        self.error = None
        self.log_path = None
        self.result = None
        self.started = None
        self.selection_started = None
        self.proc = None
        self.job = None
        self.k = 0
        self.budget = 175

    def _go_path(self, scenario):
        return os.path.join(self.data_dir, "repairs", scenario + "_go")

    def snapshot(self):
        with self.lock:
            self._refresh_phase_locked()
            if self.state == "selecting" and self.selection_started:
                elapsed = _now() - self.selection_started
            elif self.started:
                elapsed = _now() - self.started
            else:
                elapsed = 0.0
            log_tail = _log_tail(self.log_path)
            prog = _read_progress(self.data_dir, self.scenario)
            k = self.k
            if prog and prog.get("k") is not None:
                k = int(prog["k"])
            else:
                k = _parse_k(log_tail, self.k)
            self.k = k
            msg = self.message
            if prog and self.state in ("running", "selecting"):
                msg = "Selecting %s / %s" % (k, prog.get("budget") or self.budget)
            elif self.state in ("running", "queued", "preparing", "selecting") and log_tail:
                last = [ln for ln in log_tail.splitlines() if ln.strip()]
                if last:
                    msg = last[-1][:160]
            return {
                "state": self.state,
                "mode": self.mode,
                "scenario": self.scenario,
                "elapsed_s": elapsed,
                "message": msg,
                "log_tail": log_tail,
                "k": k,
                "budget": self.budget,
                "result": self.result,
                "error": self.error,
                "jobid": None if not self.job else self.job.get("jobid"),
            }

    def _refresh_phase_locked(self):
        if self.mode != "prepare":
            return
        if self.state in ("idle", "done", "error", "selecting"):
            return
        prog = _read_progress(self.data_dir, self.scenario)
        phase = (prog or {}).get("phase")
        if _log_has(self.log_path, "GO received") or phase == "selecting":
            if self.state != "selecting":
                self.state = "selecting"
                if self.selection_started is None:
                    self.selection_started = _now()
                self.message = "Selection running"
            return
        if _log_has(self.log_path, "READY_FOR_SELECTION") or phase == "ready":
            self.state = "ready"
            self.message = "Ready for selection"

    def start(self, scenario, mode="all"):
        if scenario not in ("branch_north", "branch_south"):
            return False, "Unknown scenario"
        if mode not in ("all", "prepare"):
            return False, "Unknown mode"
        launch = os.path.join(os.path.dirname(__file__), "launch_repair.sh")
        if not os.path.isfile(launch):
            return False, "Live repair is not available in this package"
        sc_path = os.path.join(self.data_dir, "scenarios", scenario + ".json")
        if not os.path.isfile(sc_path):
            return False, "Scenario JSON missing"
        k_path = os.environ.get("OED_K", K_DEFAULT)
        if not os.path.isfile(k_path):
            return False, "K matrix not found at %s (set OED_K if it lives elsewhere)" % k_path

        with self.lock:
            busy = ("running", "queued", "preparing", "ready", "selecting")
            if self.state in busy:
                if self.scenario == scenario and mode == self.mode:
                    return True, "already running"
                return False, "A repair is already running (%s)" % self.scenario
            job, err = find_slurm_job()
            if err:
                return False, "squeue failed: %s" % err
            if not job:
                return False, "No GPU allocation. Request an interactive job, then click Repair."
            self.mode = mode
            if job["state"] != "RUNNING":
                self.state = "queued"
            elif mode == "prepare":
                self.state = "preparing"
            else:
                self.state = "running"
            self.scenario = scenario
            self.message = (
                "Waiting for GPU job %s…" % job["jobid"]
                if job["state"] != "RUNNING"
                else (
                    "Loading K and factorizing…"
                    if mode == "prepare"
                    else "Launching repair on job %s" % job["jobid"]
                )
            )
            self.error = None
            self.result = None
            self.started = _now()
            self.selection_started = None
            self.job = job
            self.k = 0
            self.log_path = os.path.join(
                self.data_dir, "repairs", scenario + "_live.log"
            )
            t = threading.Thread(
                target=self._run, args=(scenario, job, launch, mode), daemon=True
            )
            t.start()
            return True, self.message

    def go(self):
        with self.lock:
            self._refresh_phase_locked()
            if self.state != "ready":
                return False, "Not ready to start selection (state=%s)" % self.state
            if not self.scenario:
                return False, "No active scenario"
            path = self._go_path(self.scenario)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write("go\n")
            self.state = "selecting"
            self.selection_started = _now()
            self.message = "Selection started"
            return True, self.message

    def _run(self, scenario, job, launch, mode="all"):
        os.makedirs(os.path.join(self.data_dir, "repairs"), exist_ok=True)
        deadline = _now() + 20 * 60
        current = job
        while current and current.get("state") != "RUNNING":
            if _now() > deadline:
                with self.lock:
                    self.state = "error"
                    self.error = "Timed out waiting for job %s" % job["jobid"]
                    self.message = self.error
                return
            with self.lock:
                self.state = "queued"
                self.message = "Waiting for GPU job %s (%s)" % (
                    current.get("jobid"),
                    current.get("state"),
                )
            time.sleep(3)
            current, _err = find_slurm_job()
            if not current:
                with self.lock:
                    self.state = "error"
                    self.error = "GPU allocation disappeared before the repair started"
                    self.message = self.error
                return

        nodes = max(1, int(current.get("nodes") or 1))
        go_file = self._go_path(scenario)
        if os.path.isfile(go_file):
            try:
                os.remove(go_file)
            except OSError:
                pass
        prog_file = os.path.join(
            self.data_dir, "repairs", scenario + "_select.txt.progress.json"
        )
        if os.path.isfile(prog_file):
            try:
                os.remove(prog_file)
            except OSError:
                pass
        cmd = [
            "bash",
            launch,
            "--scenario",
            scenario,
            "--jobid",
            current["jobid"],
            "--nodes",
            str(nodes),
            "--data",
            self.data_dir,
        ]
        if mode == "prepare":
            cmd.append("--wait_for_go")
        k_path = os.environ.get("OED_K", K_DEFAULT)
        if k_path and k_path != K_DEFAULT:
            cmd.extend(["--k", k_path])
        ref_path = os.environ.get("OED_REFERENCE")
        if ref_path:
            cmd.extend(["--reference", ref_path])
        env = os.environ.copy()
        conda = env.get("OED_CONDA") or env.get("CONDA_PREFIX")
        if not conda or not os.path.isfile(os.path.join(conda, "bin", "python")):
            conda = CONDA_ENV
        env["CONDA_PREFIX"] = conda
        env["PYTHONUNBUFFERED"] = "1"
        env["HDF5_USE_FILE_LOCKING"] = "FALSE"
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("PYTHONUSERBASE", None)

        with self.lock:
            self.job = current
            if mode == "prepare":
                if self.state not in ("ready", "selecting"):
                    self.state = "preparing"
                self.message = "Loading K on job %s (%s nodes)" % (
                    current["jobid"],
                    nodes,
                )
            else:
                self.state = "running"
                self.message = "Repair running on job %s (%s nodes)" % (
                    current["jobid"],
                    nodes,
                )

        with open(self.log_path, "ab") as logf:
            logf.write(
                (
                    "\n==== live repair %s job %s nodes %s ====\n"
                    % (scenario, current["jobid"], nodes)
                ).encode("utf-8")
            )
            logf.flush()
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    cwd=self.repo_dir,
                    env=env,
                )
            except Exception as exc:
                with self.lock:
                    self.state = "error"
                    self.error = "Failed to launch repair: %s" % exc
                    self.message = self.error
                return
            with self.lock:
                self.proc = proc
            rc = proc.wait()

        out_json = os.path.join(self.data_dir, "repairs", scenario + ".json")
        with self.lock:
            if rc == 0 and os.path.isfile(out_json):
                try:
                    self.result = json.load(open(out_json))
                except Exception as exc:
                    self.state = "error"
                    self.error = "Repair finished but JSON is unreadable: %s" % exc
                    self.message = self.error
                    return
                self.state = "done"
                wall = self.result.get("wall_seconds")
                self.message = "Repair finished in %ss" % (
                    int(wall) if wall is not None else "?"
                )
            else:
                self.state = "error"
                self.error = "Repair exited %s. See log." % rc
                self.message = self.error


def _read_progress(data_dir, scenario):
    if not data_dir or not scenario:
        return None
    path = os.path.join(data_dir, "repairs", scenario + "_select.txt.progress.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            obj = json.load(f)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _log_has(path, needle):
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 48000), os.SEEK_SET)
            raw = f.read()
    except Exception:
        return False
    return needle.encode("utf-8") in raw


def _log_tail(path, n=8):
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 24000), os.SEEK_SET)
            raw = f.read().decode("utf-8", "replace")
    except Exception:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines[-n:])


def _parse_k(text, fallback):
    if not text:
        return fallback
    found = None
    for line in text.splitlines():
        m = re.search(r"(\d+)\s*/\s*175", line)
        if m:
            found = int(m.group(1))
    return fallback if found is None else found


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, web_dir=None, data_dir=None, controller=None, **kwargs):
        self.web_dir = os.path.abspath(web_dir)
        self.data_dir = os.path.abspath(data_dir)
        self.controller = controller
        try:
            super(Handler, self).__init__(*args, directory=self.web_dir, **kwargs)
        except TypeError:
            super(Handler, self).__init__(*args, **kwargs)

    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/alloc":
            job, err = find_slurm_job()
            if err:
                self._send_json({"error": err}, 500)
                return
            if not job:
                self._send_json({"jobid": None, "state": "none"})
                return
            self._send_json(job)
            return
        if parsed.path in ("/api/repair/status", "/api/repair"):
            if self.controller is None:
                self._send_json({"state": "idle"}, 200)
                return
            self._send_json(self.controller.snapshot())
            return
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/api/repair", "/api/repair/go"):
            self.send_error(404, "No such API")
            return
        if self.controller is None:
            self._send_json({"error": "Live repair is not configured"}, 501)
            return
        try:
            body = self._read_json_body()
        except Exception:
            self._send_json({"error": "Invalid JSON body"}, 400)
            return
        if parsed.path == "/api/repair/go":
            ok, msg = self.controller.go()
            if not ok:
                self._send_json({"error": msg}, 409)
                return
            self._send_json({"ok": True, "message": msg, **self.controller.snapshot()})
            return
        scenario = (body.get("scenario") or "").strip()
        mode = (body.get("mode") or "all").strip()
        ok, msg = self.controller.start(scenario, mode=mode)
        if not ok:
            self._send_json({"error": msg}, 409 if "already" in msg else 400)
            return
        self._send_json({"ok": True, "message": msg, **self.controller.snapshot()})

    def translate_path(self, path):
        parsed = urlparse(path)
        p = unquote(parsed.path)
        if p.startswith("/data/") or p == "/data":
            rel = p[len("/data/") :] if p.startswith("/data/") else ""
            full = os.path.abspath(os.path.join(self.data_dir, rel))
            root = self.data_dir + os.sep
            if full != self.data_dir and not full.startswith(root):
                return os.path.join(self.data_dir, "")
            return full
        rel = p.lstrip("/")
        if not rel:
            rel = "index.html"
        full = os.path.abspath(os.path.join(self.web_dir, rel))
        root = self.web_dir + os.sep
        if full != self.web_dir and not full.startswith(root):
            return os.path.join(self.web_dir, "index.html")
        return full

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--web",
        default=os.path.join(os.path.dirname(__file__), "web"),
    )
    ap.add_argument(
        "--data",
        default="/pscratch/sd/s/srvenkat/oed_demo/data",
    )
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Bind address. 127.0.0.1 lets Cursor/VS Code auto-forward a clickable local URL.",
    )
    args = ap.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    controller = RepairController(args.data, repo)
    handler = functools.partial(
        Handler,
        web_dir=os.path.abspath(args.web),
        data_dir=os.path.abspath(args.data),
        controller=controller,
    )
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    url = "http://127.0.0.1:%s/" % args.port
    with socketserver.ThreadingTCPServer((args.bind, args.port), handler) as httpd:
        print("Demo UI:  %s" % url)
        print("\033]8;;%s\033\\%s\033]8;;\033\\" % (url, url))
        print("If the link does not open, use Ports → Forwarded Address for this port.")
        print("Web root: %s" % args.web)
        print("Data:     %s" % args.data)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
