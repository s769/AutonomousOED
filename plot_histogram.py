import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Plot histogram (appendix wrapper).")
    parser.add_argument("--optimal", required=True, help="Optimal checkpoint file (sensor indices + score).")
    parser.add_argument("--random", required=True, help="Random checkpoint file (one score per line).")
    parser.add_argument("--budget", type=int, default=175)
    args = parser.parse_args()

    from oed_hist import plot_histogram

    plot_histogram(args.random, args.optimal, args.budget, args)


if __name__ == "__main__":
    main()

