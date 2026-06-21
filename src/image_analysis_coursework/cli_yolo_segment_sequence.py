"""Command line entrypoint for segmenting a full microscopy sequence with YOLO."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trained YOLO segmentation model over all frames in one sequence.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sequence", default="01")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part_ii/c/labels"))
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from image_analysis_coursework.tracking import segment_sequence_with_model

    label_paths = segment_sequence_with_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        sequence=args.sequence,
        output_dir=args.output_dir,
        confidence=args.confidence,
    )
    print(f"Segmented {len(label_paths)} frames")
    for path in label_paths[:5]:
        print(f"label: {path}")
    if len(label_paths) > 5:
        print("...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
