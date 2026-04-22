# AutonomousOED (SC26 AD/AE Appendix)

This repository contains the scripts used to reproduce the performance benchmarks and (with access to the large CSZ dataset) the greedy sensor-selection application described in the SC26 paper.

## Repository layout

- `generate_dummy_data.py`: create a synthetic 2D-chunked HDF5 matrix for scaling benchmarks.
- `run_single_gpu.py`: single-GPU benchmark for naive vs Schur-update formulations.
- `scaling_benchmark.py`: multi-GPU benchmark with overlapped POSIX I/O + compute.
- `run_strong_scaling.sh`, `run_weak_scaling.sh`: Slurm wrappers for scaling runs.
- `select_sensors.py`: MPI greedy selection on an HDF5-backed CSZ \(K\) matrix (real dataset required).
- `run_cascadia_oed.py`: appendix-friendly wrapper around `select_sensors.py`.
- `eval_random_configs.py`: generate random-configuration baselines (scores only).
- `plot_histogram.py`: plot the histogram comparing optimal vs random baselines.

## Installation

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For parallel file systems, it is recommended to disable HDF5 file locking:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

## Task 1: Generate dummy data (for scaling benchmarks)

```bash
python generate_dummy_data.py --sensors 600 --timesteps 420 --output dummy_data.h5
```

## Task 2: Single-GPU benchmark (Figure 2)

```bash
python run_single_gpu.py --max_budget 400 --Nt 420 --out ablation_results_420_single.csv
```

The output CSV contains peak memory and time-per-iteration for:
- Naive formulation (out-of-place / in-place)
- Schur-update formulation (out-of-place / strict in-place)

## Task 3: Multi-GPU scaling benchmarks (Figure 3)

These wrappers assume Slurm and `srun`. Provide the HDF5 file and the maximum rank count to test.

Strong scaling:

```bash
bash run_strong_scaling.sh --data dummy_data.h5 --max-ranks 512
```

Weak scaling:

```bash
bash run_weak_scaling.sh --data dummy_data.h5 --max-ranks 512
```

Outputs are written under `scaling_results/` by default.

## Task 4: Cascadia application (Figure 5, real dataset required)

With the real HDF5 \(K\) matrix available on a parallel filesystem, run the greedy selection via MPI. Example:

```bash
srun -n 16 python run_cascadia_oed.py \
  --data cascadia_K_matrix.h5 \
  --budget 175 \
  --r_sq 1.0 \
  --out cascadia_optimal_results.txt
```

## Task 5: Random baseline + histogram (Figure 4)

Generate random baseline scores (MPI-capable; produces a single-column checkpoint file):

```bash
python eval_random_configs.py \
  --data cascadia_K_matrix.h5 \
  --samples 100 \
  --budget 175 \
  --out random_results.txt
```

Plot histogram comparing random baselines to the greedy optimum:

```bash
python plot_histogram.py \
  --optimal cascadia_optimal_results.txt \
  --random random_results.txt \
  --budget 175
```

## Notes for artifact packaging

- Large datasets (e.g., the CSZ \(K\) matrix) are intentionally **not** included in this repository.
- All scripts accept file paths via CLI arguments; there are no site-specific absolute paths in the code.
