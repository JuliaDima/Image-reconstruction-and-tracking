"""Command line entrypoint for Part II.B YOLO training."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO segmentation model for cell masks.")
    parser.add_argument("--yolo-dataset-dir", type=Path, default=Path("outputs/part_ii/b/yolo_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part_ii/b/yolo_runs"))
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from image_analysis_coursework.yolo_segmentation import train_yolo_model

    metrics = train_yolo_model(
        yolo_dataset_dir=args.yolo_dataset_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
