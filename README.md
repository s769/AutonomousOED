# AutonomousOED (SC26 AD/AE Appendix)

This repository contains the scripts used to reproduce the performance benchmarks and (with access to the large CSZ dataset) the greedy sensor-selection application described in the SC26 paper.

## Repository layout

- `create_test_mat.py`: create a synthetic 2D-chunked HDF5 matrix for scaling benchmarks.
- `single_gpu_benchmark.py`: single-GPU benchmark for naive vs Schur-update formulations.
- `scaling_benchmark.py`: multi-GPU benchmark with overlapped POSIX I/O + compute.
- `run_scaling.sh`: Slurm wrapper to run both strong + weak scaling.
- `select_sensors.py`: MPI greedy selection on an HDF5-backed CSZ \(K\) matrix (real dataset required).
- `oed_hist.py`: generate random-configuration baselines and (optionally) plot the histogram.

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
python create_test_mat.py --file dummy_data.h5 --total_candidates 600 --Nt 420 --precision double
```

## Task 2: Single-GPU benchmark (Figure 2)

```bash
python single_gpu_benchmark.py --max_budget 400 --Nt 420 --file ablation_results_420_single.csv
```

The output CSV contains peak memory and time-per-iteration for:
- Naive formulation (out-of-place / in-place)
- Schur-update formulation (out-of-place / strict in-place)

## Task 3: Multi-GPU scaling benchmarks (Figure 3)

This script assumes Slurm and `srun`. Provide the HDF5 file and the maximum rank count to test.

```bash
bash run_scaling.sh --data dummy_data.h5 --max-ranks 512
```
Outputs are written under `scaling_results/` by default (both strong and weak scaling CSVs).

## Task 4: Cascadia application (Figure 5, real dataset required)

With the real HDF5 \(K\) matrix available on a parallel filesystem, run the greedy selection via MPI. Example:

```bash
srun -n 16 python select_sensors.py cascadia_K_matrix.h5 175 \
  --r_sq 1.0 \
  --checkpoint_file cascadia_optimal_results.txt
```

## Task 5: Random baseline + histogram (Figure 4)

Generate random baseline scores (MPI-capable; produces a single-column checkpoint file). This run skips plotting:

```bash
python oed_hist.py \
  --h5_store_K cascadia_K_matrix.h5 \
  --total_samples 100 \
  --budget 175 \
  --checkpoint_file random_results.txt \
  --no_plot
```

Plot histogram comparing random baselines to the greedy optimum:

```bash
python oed_hist.py \
  --plot_only \
  --checkpoint_file random_results.txt \
  --optimal_file cascadia_optimal_results.txt \
  --budget 175
```

## Notes for artifact packaging

- Large datasets (e.g., the CSZ \(K\) matrix) are intentionally **not** included in this repository.
- All scripts accept file paths via CLI arguments; there are no site-specific absolute paths in the code.
