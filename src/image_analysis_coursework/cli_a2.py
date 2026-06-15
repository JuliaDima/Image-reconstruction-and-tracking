"""Command line entrypoint for Part I.A.2."""

from __future__ import annotations

import argparse
from pathlib import Path


def _lambda_values(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("provide at least one lambda value")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Part I.A.2 classical TV restoration experiments.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--a1-output-dir", type=Path, default=Path("outputs/part_i/a1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part_i/a2"))
    parser.add_argument("--sigma-blur", type=float, default=2.0)
    parser.add_argument("--noise-std", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambdas", type=_lambda_values, default=(0.001, 0.01, 0.08))
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Optional TIFF path. If omitted, the default dataset is downloaded.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from image_analysis_coursework.a2 import run_a2_from_image

    result, files = run_a2_from_image(
        image_path=args.image_path,
        data_dir=args.data_dir,
        a1_output_dir=args.a1_output_dir,
        a2_output_dir=args.output_dir,
        blur_sigma=args.sigma_blur,
        noise_std=args.noise_std,
        seed=args.seed,
        lambda_values=args.lambdas,
        num_iterations=args.iterations,
        gaussian_sigma=args.gaussian_sigma,
    )

    print(f"Output directory: {args.output_dir}")
    print(f"Noisy input: PSNR {result.noisy_psnr:.2f} dB, SSIM {result.noisy_ssim:.4f}")
    print(
        f"Gaussian sigma={result.gaussian_sigma:g}: "
        f"PSNR {result.gaussian_psnr:.2f} dB, SSIM {result.gaussian_ssim:.4f}"
    )
    for tv_result in result.tv_results:
        print(
            f"TV lambda={tv_result.lambda_value:g}: "
            f"PSNR {tv_result.psnr:.2f} dB, SSIM {tv_result.ssim:.4f}, "
            f"objective {tv_result.objective[0]:.3f} -> {tv_result.objective[-1]:.3f}"
        )
    for name, path in files.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
