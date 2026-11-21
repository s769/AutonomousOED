# AutonomousOED
[![DOI](https://zenodo.org/badge/1218415652.svg)](https://doi.org/10.5281/zenodo.19701232)


This repository contains the scripts used to reproduce the performance benchmarks and (with access to the large CSZ dataset) the greedy sensor-selection application described in paper: "Sensor Placement for Tsunami Early Warning via Large-Scale Bayesian Optimal Experimental Design."

## Repository layout

- `create_test_mat.py`: create a synthetic 2D-chunked HDF5 matrix for the benchmarks.
- `single_gpu_benchmark.py`: single-GPU benchmark for naive vs Schur-update formulations.
- `scaling_benchmark.py`: multi-GPU benchmark with overlapped POSIX I/O + compute.
- `run_scaling.sh`: Slurm wrapper to run both strong + weak scaling.
- `io_profile.py`: per-candidate I/O and GPU runtime breakdown.
- `select_sensors.py`: MPI greedy selection on an HDF5-backed CSZ \(K\) matrix (real dataset required).
- `oed_hist.py`: generate random-configuration baselines and plot the histogram.
- `evaluate_uniform_grid.py`: objective value of a uniform-grid sensor network.
- `plot_single_gpu.py`, `plot_scaling_results.py`, `plot_io_trace.py`, `animate_sensor_select.py`: figures.

## Installation

**PyTorch must be a GPU build that matches your system.** A plain `pip install torch` can give a
CPU-only build; the benchmarks then fail because no GPU is detected. Install PyTorch first, following
your site's instructions (for example a CUDA wheel from pytorch.org on NVIDIA systems, or a ROCm build
on AMD systems; on Frontier see the PyTorch section of the OLCF user guide), then install the rest:

```bash
python -m venv .venv
source .venv/bin/activate
# 1. install a GPU-enabled PyTorch for your system (see above)
# 2. install the remaining dependencies
pip install -r requirements.txt
```

Check that PyTorch sees the GPU before running anything:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

`requirements.txt` lists the versions the artifact was tested with (PyTorch 2.9.1, Python 3.11).
If your site provides a different PyTorch build, keep it and remove the `torch` line from
`requirements.txt` so that pip does not replace it with a CPU-only wheel.

For parallel file systems, it is recommended to disable HDF5 file locking:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

## Task 1: Generate dummy data (for the benchmarks)

```bash
python create_test_mat.py --file dummy_data.h5 --total_candidates 600 --Nt 420 --precision double
```

This writes a file of about 473 GB and **takes roughly one hour** on a parallel file system
(a progress bar is shown). It only has to be done once; Tasks 2-4 reuse the file.

## Task 2: Single-GPU benchmark (Figure 2)

```bash
python single_gpu_benchmark.py --max_budget 400 --Nt 420 --file results.csv
```

The output CSV contains peak memory and time-per-iteration for:
- Naive formulation (out-of-place / in-place)
- Schur-update formulation (out-of-place / strict in-place)

Plot it (any file name works; `--labels` sets the legend entry):

```bash
python plot_single_gpu.py results.csv --labels MI250X
```

Several GPUs can be compared with `python plot_single_gpu.py a.csv b.csv --labels A100 GH200`.
Without arguments the script plots every `single_gpu_results_<gpu>.csv` found in `scaling_results/`.
The figures `naive_performance.pdf` and `schur_performance.pdf` are written next to the first input
file (or to `--outdir`). LaTeX is used for the labels when it is installed; pass `--no_tex` to skip it.

## Task 3: Multi-GPU scaling benchmarks (Figure 3)

This script assumes Slurm and `srun`. Provide the HDF5 file and the maximum rank count to test.

```bash
bash run_scaling.sh --data dummy_data.h5 --max-ranks 512
```

Outputs are written under `scaling_results/` by default as `pm_strong_scaling.csv` and
`pm_weak_scaling.csv`. Use `--prefix <name>` to change the `pm` prefix (for example `--prefix fr`
on Frontier) and `--ranks-per-node` to match your system (4 on Perlmutter, 8 on Frontier, where
each MI250X exposes two devices).

Plot whatever is there; data from a single system is enough:

```bash
python plot_scaling_results.py --data_dir scaling_results --annotate
```

The prefixes `pm` and `fr` are labelled Perlmutter and Frontier. For another system use, for example,
`--systems mysys=MyCluster --ranks_per_node 4`.

## Task 4: Execution timeline trace (Figure 4)

Record the pipelined I/O / compute timeline of rank 0 during a 16-GPU run. `--timeline` takes the
output file (equivalently `--trace_file <file>`):

```bash
srun -N 4 -n 16 python scaling_benchmark.py \
  --h5_path dummy_data.h5 \
  --file timeline_run.csv \
  --total_candidates 4096 \
  --runs 1 \
  --timeline io_overlap_timeline.json \
  --timeline_record_from 150 \
  --timeline_num_candidates 5
```

Measure the GPU runtime breakdown of one candidate:

```bash
srun -n 1 python io_profile.py \
  --h5_path dummy_data.h5 \
  --breakdown --breakdown_candidate 150 \
  --breakdown_file io_candidate_breakdown.json
```

Plot both:

```bash
python plot_io_trace.py \
  --trace_file io_overlap_timeline.json \
  --breakdown_file io_candidate_breakdown.json \
  --output overlap_trace.pdf
```

`io_profile.py --timeline` records the same kind of timeline on a single GPU; there `--timeline` is a
switch and the output file is given with `--trace_file` (`--timeline <file>` is also accepted).

## Task 5: Cascadia application (Figure 6, real dataset required)

The dataset (about 500 GB) is hosted on [HuggingFace](https://doi.org/10.57967/hf/10077). It is stored
with Git LFS, so **`git lfs pull` is required**: a plain `git clone` only fetches small pointer files.

```bash
git lfs install                      # once per machine; on some systems: module load git-lfs
git clone https://huggingface.co/datasets/s769/cascadia_K_matrix
cd cascadia_K_matrix
git lfs pull                         # downloads the matrix parts; this can take a couple of hours
cat cascadia_K_matrix.h5.part* > cascadia_K_matrix.h5
```

With the real HDF5 \(K\) matrix available on a parallel filesystem, run the greedy selection via MPI. Example:

```bash
srun -n 16 python select_sensors.py cascadia_K_matrix.h5 175 \
  --r_sq 1.0 \
  --checkpoint_file cascadia_optimal_results.txt
```

Map and animation of the selection:

```bash
python animate_sensor_select.py --sensors 600_sensor_coords.csv --indices cascadia_optimal_results.txt
```

## Task 6: Random baseline, uniform grid and histogram (Figure 5)

Generate random baseline scores (MPI-capable; produces a single-column checkpoint file). This run skips plotting:

```bash
python oed_hist.py \
  --h5_store_K cascadia_K_matrix.h5 \
  --total_samples 100 \
  --budget 175 \
  --checkpoint_file random_results.txt \
  --no_plot
```

Evaluate the uniform-grid network (one GPU; writes the value to `uniform_result.txt`):

```bash
python evaluate_uniform_grid.py \
  --h5_store_K cascadia_K_matrix.h5 \
  --budget 175 \
  --coords_file 600_sensor_coords.csv
```

The factor is built block by block, so the peak GPU memory is a single
\((B N_t) \times (B N_t)\) matrix (about 22 GB in single precision for \(B = 175\)).

Plot the histogram comparing random baselines to the greedy optimum and the uniform grid:

```bash
python oed_hist.py \
  --plot_only \
  --checkpoint_file random_results.txt \
  --optimal_file cascadia_optimal_results.txt \
  --uniform_file uniform_result.txt \
  --budget 175
```

`--optimal_value` and `--uniform_value` can be used instead of the two files to pass the numbers directly.

## Citation

If you use this software, please cite the paper (GitHub's "Cite this repository" button reads `CITATION.cff`):

```bibtex
@inproceedings{venkat2026sensor,
  author    = {Venkat, Sreeram and Henneking, Stefan and Ghattas, Omar},
  title     = {Sensor placement for tsunami early warning via large-scale {Bayesian} optimal experimental design},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  series    = {SC '26},
  year      = {2026},
  pages     = {234--253},
  doi       = {10.1109/SC51908.2026.00022}
}
```

## Notes for artifact packaging

- Large datasets (e.g., the CSZ \(K\) matrix) are intentionally **not** included in this repository.
- All scripts accept file paths via CLI arguments.
