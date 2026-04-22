import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate random configurations and save histogram input (appendix wrapper)."
    )
    parser.add_argument("--data", required=True, help="Path to HDF5 K matrix.")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--budget", type=int, default=175)
    parser.add_argument("--out", default="random_results.txt", help="Output checkpoint file.")
    parser.add_argument(
        "--optimal",
        default=None,
        help="Optional optimal checkpoint (only needed if you want this wrapper to also plot).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r_sq", type=float, default=1.0)
    parser.add_argument("--precision", choices=["single", "double"], default="single")
    parser.add_argument("--exclude_indices", default="")
    parser.add_argument("--plot", action="store_true", help="Also plot histogram at the end.")
    args = parser.parse_args()

    argv = [
        "oed_hist.py",
        "--h5_store_K",
        args.data,
        "--budget",
        str(args.budget),
        "--total_samples",
        str(args.samples),
        "--seed",
        str(args.seed),
        "--r_sq",
        str(args.r_sq),
        "--precision",
        args.precision,
        "--checkpoint_file",
        args.out,
    ]
    if args.plot:
        if not args.optimal:
            raise ValueError("--optimal is required when --plot is set.")
        argv += ["--optimal_file", args.optimal]
    else:
        argv += ["--no_plot"]
    if args.exclude_indices:
        argv += ["--exclude_indices", args.exclude_indices]

    # We don't want plotting here; this wrapper is "compute-only".
    from oed_hist import main as oed_main

    sys.argv = argv
    oed_main()


if __name__ == "__main__":
    main()

