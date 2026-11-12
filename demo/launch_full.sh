#!/usr/bin/env bash
# Load the cluster stack, then run demo/run_full.py against a held allocation.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  . /etc/profile.d/modules.sh
fi
set +u
module load PrgEnv-gnu cray-hdf5-parallel >/dev/null 2>&1 || true
set -u

if [[ -n "${OED_CONDA:-}" ]]; then
  export CONDA_PREFIX="${OED_CONDA}"
elif [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
  export CONDA_PREFIX="/global/common/software/m4767/srvenkat/conda/envs/h5torch-mpi-cu13"
fi
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
unset PYTHONUSERBASE || true
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

exec python -u demo/run_full.py "$@"
