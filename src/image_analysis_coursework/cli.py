"""Command line entrypoints for coursework tasks."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Part I.A.1 forward image acquisition simulation.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part_i/a1"))
    parser.add_argument("--sigma-blur", type=float, default=2.0)
    parser.add_argument("--noise-std", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Optional TIFF path. If omitted, the default dataset is downloaded.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from image_analysis_coursework.a1 import (
        find_default_ground_truth,
        load_ground_truth,
        save_a1_outputs,
        simulate_acquisition,
    )

    image_path = args.image_path or find_default_ground_truth(args.data_dir)
    ground_truth = load_ground_truth(image_path)
    result = simulate_acquisition(
        ground_truth,
        blur_sigma=args.sigma_blur,
        noise_std=args.noise_std,
        seed=args.seed,
        source_image=image_path,
    )
    files = save_a1_outputs(result, args.output_dir)

    print(f"Source image: {image_path}")
    print(f"Output directory: {args.output_dir}")
    for name, path in files.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
