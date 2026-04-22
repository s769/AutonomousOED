import argparse

from create_test_mat import create_dummy_h5


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic, 2D-chunked HDF5 data for scaling benchmarks."
    )
    parser.add_argument("--output", required=True, help="Output HDF5 file path.")
    parser.add_argument("--sensors", type=int, default=600, help="Number of candidates (Nd).")
    parser.add_argument("--timesteps", type=int, default=420, help="Temporal block size (Nt).")
    parser.add_argument("--precision", choices=["single", "double"], default="double")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    create_dummy_h5(
        filename=args.output,
        total_candidates=args.sensors,
        Nt=args.timesteps,
        precision=args.precision,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
