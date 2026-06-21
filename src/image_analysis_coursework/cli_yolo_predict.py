"""Command line entrypoint for Part II.B YOLO prediction overlay."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlay YOLO segmentation predictions on one image.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True, nargs="+")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/part_ii/b/prediction_overlay.png"))
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from image_analysis_coursework.yolo_segmentation import overlay_prediction, save_prediction_montage

    if len(args.image_path) == 1:
        output = overlay_prediction(args.model_path, args.image_path[0], args.output_path, confidence=args.confidence)
    else:
        output = save_prediction_montage(args.model_path, args.image_path, args.output_path, confidence=args.confidence)
    print(f"overlay: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
