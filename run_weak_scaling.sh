#!/bin/bash
set -e

usage() {
  cat <<'EOF'
Usage:
  bash run_weak_scaling.sh --data <dummy_data.h5> --max-ranks <N> [options]

Options:
  --data <path>              Path to HDF5 file (required)
  --max-ranks <N>            Maximum ranks to test (required)
  --out <csv>                Output CSV (default: scaling_results/weak_scaling.csv)
  --weak-per-rank <N>        Candidates per rank (default: 256)
  --runs <N>                 Repetitions per point (default: 3)
  --start-ranks <N>          Starting ranks (default: 4)
  --ranks-per-node <N>       Used to estimate nodes for srun (default: 4)
  --extra-srun "<args...>"   Extra args passed to srun (default: "")
EOF
}

H5_PATH=""
MAX_RANKS=""
OUT="scaling_results/weak_scaling.csv"
WEAK_PER_RANK=256
RUNS=3
START_RANKS=4
RANKS_PER_NODE=4
EXTRA_SRUN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) H5_PATH="$2"; shift 2 ;;
    --max-ranks) MAX_RANKS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --weak-per-rank) WEAK_PER_RANK="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --start-ranks) START_RANKS="$2"; shift 2 ;;
    --ranks-per-node) RANKS_PER_NODE="$2"; shift 2 ;;
    --extra-srun) EXTRA_SRUN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$H5_PATH" || -z "$MAX_RANKS" ]]; then
  usage
  exit 2
fi

mkdir -p "$(dirname "$OUT")"

for (( ranks=START_RANKS; ranks<=MAX_RANKS; ranks*=2 )); do
  total_cand=$(( WEAK_PER_RANK * ranks ))
  nodes=$(( ranks / RANKS_PER_NODE ))
  if [ $nodes -eq 0 ]; then
    nodes=1
  fi
  srun -N "$nodes" -n "$ranks" -u $EXTRA_SRUN python scaling_benchmark.py \
      --h5_path "$H5_PATH" \
      --file "$OUT" \
      --total_candidates "$total_cand" \
      --runs "$RUNS"
done

