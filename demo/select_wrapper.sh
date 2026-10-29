#!/usr/bin/env bash
# Per-rank CUDA + Torch lib path for Perlmutter (NERSC CUDA 13 vs pip torch).
set -euo pipefail
if [[ -n "${SLURM_LOCALID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_LOCALID}"
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  # python3.12 (new cu13 env) or python3.11 (legacy)
  for sp in "${CONDA_PREFIX}/lib/python3.12/site-packages/torch/lib" \
            "${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib"; do
    if [[ -d "$sp" ]]; then
      export LD_LIBRARY_PATH="${sp}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
      break
    fi
  done
fi
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  exec "${CONDA_PREFIX}/bin/python" -u "$@"
fi
exec python -u "$@"
