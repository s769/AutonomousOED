# Cascadia OED live demo (SC26)

Interactive HTML demo: reuse a precomputed covariance `K` to **adapt** a
175-sensor D-optimal network when a cable branch becomes unavailable.

**Message.** Expensive physics calculations can be reused to adapt a
high-fidelity observing system when deployment constraints change.

## Layout

| Path | Role |
|------|------|
| `demo/web/` | UI (Leaflet, bundled; no CDN) |
| `demo/web/vendor/` | Local Leaflet 1.9.4 |
| `demo/serve_demo.py` | Serves UI; `/data` → scratch |
| `demo/run_repair.py` | GPU repair (`--exclude_file` + restart) |
| `scripts/demo_build_web_assets.py` | Sensors, scenarios, reference UQ PNGs |
| `scripts/demo_eval_objectives.py` | Independent slogdet of original/remainder/repair |
| `scripts/demo_package_portable.py` | Offline talk folder (no `K` matrix) |
| `/pscratch/sd/s/srvenkat/oed_demo/data/` | Heavy assets (not `$HOME`, not git) |

## Conventions (do not mix datasets)

- **Sensors:** IDs `0…599` from `mw-mesh-300m-sens-12x50-latlon.csv` (`sensors.json`).
- **K:** `/pscratch/sd/s/skw/CDS_SHARED/mw-mesh-420s/files/K_prior_4_2d_rebuilt.h5`,
  `Nt = 420`, `K = I + r_sq C` with `r_sq = 1e4`. After this scaling, `R = I`.
- **Objective:** greedy maximises `logdet(K_S)`. Talk numbers use the noise-normalized
  expected information gain `f(S) = 0.5 logdet(K_S)` (nats). There is **no** leftover
  `n log σ²` term when comparing 147/166 vs 175 sensors.
- **Reference design:** `testing/certificates/oed175_rebuilt_select_single_n4.txt`.
- **UQ rasters:** posterior **standard deviation of seafloor displacement (m)** from
  `oed175_rebuilt_disp_uq_anim.h5` (176 frames: empty + 175). Same rebuilt `K`,
  same `r_sq`, same selection file. This field is a **diagnostic**, not the
  D-optimal objective. Remainder/repair views **hide** it (no matching rasters).

## Development serve (Perlmutter / Cursor)

```bash
python demo/serve_demo.py --port 8765
# http://127.0.0.1:8765/  — Ports panel if the link does not open
```

## Portable talk package (no cluster)

```bash
python scripts/demo_package_portable.py
python /pscratch/sd/s/srvenkat/oed_demo/portable_sc26/serve.py --port 8765
```

Copy `portable_sc26/` to a laptop. Requires only Python 3.6+ (3.7+ preferred).
Leaflet, basemap, PNGs, and metrics are local. **Do not** copy `K_*.h5`.

## Repair jobs (already run)

```bash
module load PrgEnv-gnu cray-hdf5-parallel conda
conda activate /global/common/software/m4767/srvenkat/conda/envs/h5torch-mpi-cu13
python demo/run_repair.py --scenario branch_north --jobid $SLURM_JOB_ID --nodes 2 --nodelist <80GB,80GB>
python demo/run_repair.py --scenario branch_south --jobid $SLURM_JOB_ID --nodes 2 --nodelist <80GB,80GB>
```

Last successful run: job `60119123`, **8× A100 80GB**, env `h5torch-mpi-cu13`.

| Scenario | Retained | Replaced | Wall clock (rebuild + fill) |
|----------|----------|----------|-----------------------------|
| North    | 166      | 9        | 214 s                       |
| South    | 147      | 28       | 392 s                       |

## Presenter script (~2 minutes)

1. **Physics is offline.** “We computed the wave-propagation operators once.
   Design only reads `K`.” Play ~2 s of the sequence, then **Jump to 175**.
   The color field is displacement uncertainty (metres), a diagnostic — the
   design criterion is D-optimal information gain about the parameters.
2. **Ask the audience:** north or south cable corridor unavailable (survey,
   not an earthquake, not sequential inference). State the one-liner:
   “South branch unavailable: 28 selected locations must be replaced.”
3. **Repair the network.** Gray = kept, teal = replacements, × = lost.
   “Keep feasible sensors; greedy-fill back to 175.” Either click
   **Repair network** (load + factorize + greedy fill in one shot), or
   **Load & factorize** first, then **Start selection** when you want the
   loop to run. The headline reports how many replacements were chosen and
   how long the selection loop took.
4. **Compare remainder vs repaired.** Replacements recover Δf nats of
   information gain. The repaired network is ~99% of the original greedy
   *f(S)* even though the original design is infeasible under the exclusion.

Reset demo between takes. Full screen for the projector.

## Missing / not in the talk UI

- Displacement-UQ rasters for remainder and repaired networks (would need the
  adjoint UQ pipeline on those sensor sets). Overlay is hidden, not faked.
- An optimality **certificate** for the repairs. The original-network ~87%
  bound does not apply to this constrained completion class.
- Remainder displacement-UQ rasters and a repair optimality certificate remain
  omitted (see `MISSING.md`). Independent GPU Cholesky of *f(S)* and repaired-network
  UQ frames completed on job 60127421.
