"""Command line entrypoint for Part II.C tracking."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track cells from a sequence of label images.")
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part_ii/c"))
    parser.add_argument("--first-frame-path", type=Path, default=None)
    parser.add_argument("--max-distance", type=float, default=35.0)
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--no-laptrack", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from image_analysis_coursework.tracking import run_tracking_from_labels

    tracks, files = run_tracking_from_labels(
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        first_frame_path=args.first_frame_path,
        max_distance=args.max_distance,
        max_gap=args.max_gap,
        use_laptrack=not args.no_laptrack,
    )
    print(f"Track points: {len(tracks)}")
    for name, path in files.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
