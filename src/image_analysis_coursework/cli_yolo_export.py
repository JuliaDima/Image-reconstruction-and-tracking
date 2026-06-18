"""Command line entrypoint for Part II.B YOLO export."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export PhC-C2DH-U373 segmentation masks to YOLO segmentation format.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part_ii/b/yolo_dataset"))
    parser.add_argument("--verify-index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from image_analysis_coursework.yolo_segmentation import export_phc_dataset_to_yolo, verify_yolo_export

    records = export_phc_dataset_to_yolo(data_dir=args.data_dir, output_dir=args.output_dir)
    print(f"Exported {len(records)} labelled frames to {args.output_dir}")
    if records:
        record = records[min(args.verify_index, len(records) - 1)]
        overlay = verify_yolo_export(record, Path(args.output_dir) / "verify_overlay.png")
        print(f"verify_overlay: {overlay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
