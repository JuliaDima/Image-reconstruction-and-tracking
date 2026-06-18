"""Command line entrypoint for Part II.A optical flow."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Horn-Schunck optical flow on a consecutive PhC-C2DH-U373 pair.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part_ii/a"))
    parser.add_argument("--sequence", default="01")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--stride", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from image_analysis_coursework.motion import run_motion_experiment

    result, files = run_motion_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sequence=args.sequence,
        index=args.index,
        alpha=args.alpha,
        num_iterations=args.iterations,
        stride=args.stride,
    )
    print(f"Initial energy: {result.energy[0]:.6g}")
    print(f"Final energy: {result.energy[-1]:.6g}")
    print(f"Mean flow: u={result.u.mean():.4g}, v={result.v.mean():.4g}")
    for name, path in files.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
