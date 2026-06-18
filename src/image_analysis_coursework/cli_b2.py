"""Command line entrypoint for Part I.B(ii)."""

from __future__ import annotations

import argparse
from pathlib import Path


def _optional_int(value: str) -> int | None:
    if value.lower() in {"none", "all", ""}:
        return None
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate the Part I.B(ii) OOD unrolled PGD model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part_i/b2"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--n-iter", type=int, default=6)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--noise-var", type=float, default=0.001)
    parser.add_argument("--max-train-images", type=_optional_int, default=None)
    parser.add_argument("--max-test-images", type=_optional_int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from image_analysis_coursework.unrolling import run_b2_training

    history, evaluation, files = run_b2_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        n_iter=args.n_iter,
        features=args.features,
        learning_rate=args.learning_rate,
        noise_var=args.noise_var,
        max_train_images=args.max_train_images,
        max_test_images=args.max_test_images,
        device_name=args.device,
        seed=args.seed,
    )
    print(f"Final train loss: {history.epochs[-1]['loss']:.6f}")
    print(f"OOD test input: PSNR {evaluation.blur_psnr:.2f} dB, SSIM {evaluation.blur_ssim:.4f}")
    print(f"OOD reconstruction: PSNR {evaluation.recon_psnr:.2f} dB, SSIM {evaluation.recon_ssim:.4f}")
    for name, path in files.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
