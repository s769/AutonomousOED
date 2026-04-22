import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm


def create_dummy_h5(filename, total_candidates, Nt, precision, seed):
    dtype = np.float64 if precision == "double" else np.float32

    rows = total_candidates * Nt
    cols = total_candidates * Nt

    print("--- Generating 2D Chunked HDF5 Dummy Data ---")
    print(f"File:   {filename}")
    print(f"Shape:  ({rows}, {cols})")
    print(f"Chunks: ({Nt}, {Nt})")
    print(f"Dtype:  {dtype.__name__}")

    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = rows * cols * bytes_per_element
    print(f"Estimated File Size: {total_bytes / (1024**3):.2f} GB")
    print("-" * 45)

    out_dir = os.path.dirname(os.path.abspath(filename))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(seed)
    with h5py.File(filename, "w") as f:
        dset = f.create_dataset(
            "K_matrix", shape=(rows, cols), chunks=(Nt, Nt), dtype=dtype
        )

        for col_idx in tqdm(range(total_candidates), desc="Writing sensor columns"):
            col_start = col_idx * Nt
            col_end = col_start + Nt
            dummy_data = rng.randn(rows, Nt).astype(dtype)
            dset[:, col_start:col_end] = dummy_data

    print(f"\nSuccess! Dummy matrix saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy 2D-chunked HDF5 matrix")
    parser.add_argument(
        "--file", type=str, required=True, help="Output path (e.g., ./dummy_K.h5)"
    )
    parser.add_argument(
        "--total_candidates", type=int, default=600, help="Total number of sensors"
    )
    parser.add_argument("--Nt", type=int, default=420, help="Temporal block size")
    parser.add_argument(
        "--precision", type=str, choices=["single", "double"], default="double"
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    create_dummy_h5(
        filename=args.file,
        total_candidates=args.total_candidates,
        Nt=args.Nt,
        precision=args.precision,
        seed=args.seed,
    )