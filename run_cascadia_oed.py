import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="MPI greedy sensor selection runner (appendix wrapper)."
    )
    parser.add_argument("--data", required=True, help="Path to HDF5 K matrix.")
    parser.add_argument("--budget", type=int, default=175)
    parser.add_argument("--r_sq", type=float, required=True, help="Scaling factor r^2.")
    parser.add_argument("--out", default="checkpoint.txt", help="Output checkpoint file.")
    parser.add_argument("--restart_from", default=None, help="Restart from an existing checkpoint file.")
    parser.add_argument("--precision", choices=["single", "double"], default="double")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    argv = [
        "select_sensors.py",
        args.data,
        str(args.budget),
        "--r_sq",
        str(args.r_sq),
        "--checkpoint_file",
        args.out,
        "--precision",
        args.precision,
    ]
    if args.restart_from:
        argv += ["--restart_from", args.restart_from]
    if args.verbose:
        argv += ["--verbose"]

    from select_sensors import main as select_main

    sys.argv = argv
    select_main()


if __name__ == "__main__":
    main()
