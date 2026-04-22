#!/bin/bash
set -e

usage() {
  cat <<'EOF'
Usage:
  bash run_scaling.sh --data <dummy_or_real.h5> --max-ranks <N> [options]

Options:
  --data <path>              Path to HDF5 file containing dataset (default: $H5_PATH)
  --max-ranks <N>            Maximum ranks to test (required)
  --outdir <dir>             Output directory (default: scaling_results)
  --strong-total <N>         Total candidates for strong scaling (default: 65536)
  --weak-per-rank <N>        Candidates per rank for weak scaling (default: 256)
  --runs <N>                 Repetitions per point (default: 3)
  --start-ranks <N>          Starting ranks (default: 4)
  --ranks-per-node <N>       Used to estimate nodes for srun (default: 4)
  --extra-srun "<args...>"   Extra args passed to srun (default: "")
EOF
}

H5_PATH="${H5_PATH:-}"
OUTDIR="scaling_results"
STRONG_TOTAL=65536
WEAK_PER_RANK=256
RUNS=3
START_RANKS=4
RANKS_PER_NODE=4
EXTRA_SRUN=""
MAX_RANKS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) H5_PATH="$2"; shift 2 ;;
    --max-ranks) MAX_RANKS="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --strong-total) STRONG_TOTAL="$2"; shift 2 ;;
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

# Recommended for parallel filesystems + h5py POSIX access
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

echo "Starting MPI scaling benchmark..."

mkdir -p "$OUTDIR"


# ==========================================
# 1. STRONG SCALING (Fixed Work: 65,536)
# ==========================================

echo "--- Running Strong Scaling ---"
for (( ranks=START_RANKS; ranks<=MAX_RANKS; ranks*=2 )); do
    echo "Running with $ranks ranks..." 
    nodes=$(( ranks / RANKS_PER_NODE ))
    if [ $nodes -eq 0 ]; then
        nodes=1
    fi
    srun -N "$nodes" -n "$ranks" -u $EXTRA_SRUN python scaling_benchmark.py \
        --h5_path "$H5_PATH" \
        --file "$OUTDIR/strong_scaling.csv" \
        --total_candidates "$STRONG_TOTAL" \
        --runs "$RUNS" \
        --max_evals 0
done



# ==========================================
# 2. WEAK SCALING (Fixed Work/GPU: 256)
# ==========================================

echo "--- Running Weak Scaling ---"
for (( ranks=START_RANKS; ranks<=MAX_RANKS; ranks*=2 )); do
    echo "Running with $ranks ranks..."
    total_cand=$(( WEAK_PER_RANK * ranks ))
    nodes=$(( ranks / RANKS_PER_NODE ))
    if [ $nodes -eq 0 ]; then
        nodes=1
    fi
    srun -N "$nodes" -n "$ranks" -u $EXTRA_SRUN python scaling_benchmark.py \
        --h5_path "$H5_PATH" \
        --file "$OUTDIR/weak_scaling.csv" \
        --total_candidates "$total_cand" \
        --runs "$RUNS"
done


# ==========================================

echo "Scaling benchmark complete!"