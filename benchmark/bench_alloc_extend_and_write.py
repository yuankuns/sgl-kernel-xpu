"""B8 prefill: compare SGLang main's split path with fused SYCL allocation."""

import argparse

from bench_memory_compare import MEMORY_SETUP_PARSER, compare_extend


def main():
    parser = argparse.ArgumentParser(description=__doc__, parents=[MEMORY_SETUP_PARSER])
    parser.add_argument("--samples", type=int, default=31)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    compare_extend(8, 4096, 64, args.samples, fused=True)


if __name__ == "__main__":
    main()
