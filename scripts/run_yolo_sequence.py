#!/usr/bin/env python
"""Segment a full sequence using a trained YOLO model."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_analysis_coursework.cli_yolo_segment_sequence import main


if __name__ == "__main__":
    raise SystemExit(main())
