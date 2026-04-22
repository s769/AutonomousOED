import argparse

from single_gpu_benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Single-GPU benchmark runner (appendix wrapper).")
    parser.add_argument("--max_budget", type=int, default=400)
    parser.add_argument("--Nt", type=int, default=420)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out", type=str, default="ablation_results_420_single.csv")
    args = parser.parse_args()

    run_benchmark(
        filename=args.out,
        Nt=args.Nt,
        max_budget=args.max_budget,
        step=args.step,
        runs=args.runs,
    )


if __name__ == "__main__":
    main()
