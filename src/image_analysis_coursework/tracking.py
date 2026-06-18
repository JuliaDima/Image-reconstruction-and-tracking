"""Part II.C cell-centroid extraction and tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, measure

from image_analysis_coursework.a1 import download_dataset
from image_analysis_coursework.yolo_segmentation import segment_image


@dataclass(frozen=True)
class Detection:
    frame: int
    label: int
    y: float
    x: float
    area: float


@dataclass(frozen=True)
class TrackPoint:
    track_id: int
    frame: int
    y: float
    x: float
    detection_label: int


def extract_centroids(label_image: np.ndarray, frame_index: int) -> list[Detection]:
    detections: list[Detection] = []
    for prop in measure.regionprops(np.asarray(label_image, dtype=np.int32)):
        y, x = prop.centroid
        detections.append(Detection(frame=frame_index, label=int(prop.label), y=float(y), x=float(x), area=float(prop.area)))
    return detections


def extract_centroids_from_sequence(label_images: Iterable[np.ndarray]) -> list[Detection]:
    detections: list[Detection] = []
    for frame_index, label_image in enumerate(label_images):
        detections.extend(extract_centroids(label_image, frame_index))
    return detections


def nearest_neighbour_tracks(detections: list[Detection], max_distance: float = 35.0, max_gap: int = 1) -> list[TrackPoint]:
    by_frame: dict[int, list[Detection]] = {}
    for detection in detections:
        by_frame.setdefault(detection.frame, []).append(detection)

    active: dict[int, TrackPoint] = {}
    active_last_seen: dict[int, int] = {}
    next_track_id = 1
    points: list[TrackPoint] = []

    for frame in sorted(by_frame):
        available = list(by_frame[frame])
        assigned: set[int] = set()
        for track_id, previous in list(active.items()):
            if frame - active_last_seen[track_id] > max_gap + 1:
                active.pop(track_id, None)
                active_last_seen.pop(track_id, None)
                continue
            if not available:
                continue
            distances = [np.hypot(det.y - previous.y, det.x - previous.x) for det in available]
            best_index = int(np.argmin(distances))
            if distances[best_index] <= max_distance:
                detection = available.pop(best_index)
                point = TrackPoint(track_id=track_id, frame=frame, y=detection.y, x=detection.x, detection_label=detection.label)
                points.append(point)
                active[track_id] = point
                active_last_seen[track_id] = frame
                assigned.add(track_id)
        for detection in available:
            track_id = next_track_id
            next_track_id += 1
            point = TrackPoint(track_id=track_id, frame=frame, y=detection.y, x=detection.x, detection_label=detection.label)
            points.append(point)
            active[track_id] = point
            active_last_seen[track_id] = frame
    return points


def track_centroids(detections: list[Detection], max_distance: float = 35.0, max_gap: int = 1, use_laptrack: bool = True) -> list[TrackPoint]:
    if use_laptrack:
        try:
            import pandas as pd
            from laptrack import LapTrack

            frame = pd.DataFrame([d.__dict__ for d in detections])
            if frame.empty:
                return []
            tracker = LapTrack(
                track_dist_metric="sqeuclidean",
                splitting_dist_metric=None,
                merging_dist_metric=None,
                cutoff=max_distance**2,
                gap_closing_max_frame_count=max_gap,
                gap_closing_cutoff=max_distance**2,
            )
            tracked, _, _ = tracker.predict_dataframe(frame, coordinate_cols=["y", "x"], frame_col="frame")
            return [
                TrackPoint(
                    track_id=int(row["track_id"]),
                    frame=int(row["frame"]),
                    y=float(row["y"]),
                    x=float(row["x"]),
                    detection_label=int(row["label"]),
                )
                for _, row in tracked.iterrows()
                if int(row["track_id"]) >= 0
            ]
        except Exception:
            pass
    return nearest_neighbour_tracks(detections, max_distance=max_distance, max_gap=max_gap)


def segment_sequence_with_model(
    model_path: str | Path,
    data_dir: str | Path = "data",
    sequence: str = "01",
    output_dir: str | Path = "outputs/part_ii/c/labels",
) -> list[Path]:
    dataset_root = download_dataset(data_dir)
    frames = sorted((dataset_root / sequence).glob("*.tif"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    label_paths: list[Path] = []
    for frame in frames:
        labels = segment_image(model_path, frame)
        label_path = output_path / frame.with_suffix(".tif").name
        io.imsave(label_path, labels)
        label_paths.append(label_path)
    return label_paths


def load_label_sequence(labels_dir: str | Path) -> list[np.ndarray]:
    return [io.imread(path) for path in sorted(Path(labels_dir).glob("*.tif"))]


def save_tracking_outputs(
    tracks: list[TrackPoint],
    detections: list[Detection],
    output_dir: str | Path = "outputs/part_ii/c",
    first_frame: np.ndarray | None = None,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    files = {
        "tracks": output_path / "tracks.csv",
        "detections": output_path / "detections.csv",
        "metadata": output_path / "metadata.json",
    }
    _write_csv(files["detections"], detections, ["frame", "label", "y", "x", "area"])
    _write_csv(files["tracks"], tracks, ["track_id", "frame", "y", "x", "detection_label"])
    files["metadata"].write_text(
        json.dumps(
            {
                "num_detections": len(detections),
                "num_track_points": len(tracks),
                "num_tracks": len({point.track_id for point in tracks}),
                "tracker": "laptrack if installed, nearest-neighbour fallback otherwise",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if first_frame is not None:
        files["overlay"] = output_path / "tracks_overlay.png"
        save_track_overlay(first_frame, tracks, files["overlay"])
    return files


def run_tracking_from_labels(
    labels_dir: str | Path,
    output_dir: str | Path = "outputs/part_ii/c",
    first_frame_path: str | Path | None = None,
    max_distance: float = 35.0,
    max_gap: int = 1,
    use_laptrack: bool = True,
) -> tuple[list[TrackPoint], dict[str, Path]]:
    label_images = load_label_sequence(labels_dir)
    detections = extract_centroids_from_sequence(label_images)
    tracks = track_centroids(detections, max_distance=max_distance, max_gap=max_gap, use_laptrack=use_laptrack)
    first_frame = io.imread(first_frame_path) if first_frame_path is not None else None
    files = save_tracking_outputs(tracks, detections, output_dir=output_dir, first_frame=first_frame)
    return tracks, files


def save_track_overlay(first_frame: np.ndarray, tracks: list[TrackPoint], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    axis.imshow(first_frame, cmap="gray")
    by_track: dict[int, list[TrackPoint]] = {}
    for point in tracks:
        by_track.setdefault(point.track_id, []).append(point)
    for track_id, points in by_track.items():
        ordered = sorted(points, key=lambda point: point.frame)
        xs = [point.x for point in ordered]
        ys = [point.y for point in ordered]
        axis.plot(xs, ys, marker="o", markersize=2, linewidth=1, label=str(track_id) if track_id <= 10 else None)
    axis.set_title("Cell tracks")
    axis.axis("off")
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _write_csv(path: Path, rows: list[object], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})
