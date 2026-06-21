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
from image_analysis_coursework.yolo_segmentation import load_yolo_model, segment_image_with_model


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


def _track_centroids_with_backend(
    detections: list[Detection],
    max_distance: float,
    max_gap: int,
    use_laptrack: bool,
) -> tuple[list[TrackPoint], str]:
    if not use_laptrack:
        return nearest_neighbour_tracks(detections, max_distance=max_distance, max_gap=max_gap), "nearest-neighbour"
    try:
        import pandas as pd
        from laptrack import LapTrack
    except ImportError:
        return nearest_neighbour_tracks(detections, max_distance=max_distance, max_gap=max_gap), "nearest-neighbour"

    frame = pd.DataFrame([d.__dict__ for d in detections])
    if frame.empty:
        return [], "laptrack"
    tracker = LapTrack(
        metric="sqeuclidean",
        cutoff=max_distance**2,
        gap_closing_metric="sqeuclidean",
        gap_closing_max_frame_count=max_gap,
        gap_closing_cutoff=max_distance**2,
        splitting_cutoff=False,
        merging_cutoff=False,
    )
    tracked, _, _ = tracker.predict_dataframe(frame, coordinate_cols=["y", "x"], frame_col="frame")
    points = [
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
    return points, "laptrack"


def track_centroids(detections: list[Detection], max_distance: float = 35.0, max_gap: int = 1, use_laptrack: bool = True) -> list[TrackPoint]:
    tracks, _ = _track_centroids_with_backend(detections, max_distance, max_gap, use_laptrack)
    return tracks


def segment_sequence_with_model(
    model_path: str | Path,
    data_dir: str | Path = "data",
    sequence: str = "01",
    output_dir: str | Path = "outputs/part_ii/c/labels",
    confidence: float = 0.25,
) -> list[Path]:
    dataset_root = download_dataset(data_dir)
    frames = sorted((dataset_root / sequence).glob("*.tif"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    label_paths: list[Path] = []
    model = load_yolo_model(model_path)
    for frame in frames:
        labels = segment_image_with_model(model, frame, confidence=confidence)
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
    tracker_backend: str = "unknown",
    max_distance: float = 35.0,
    max_gap: int = 1,
    num_frames: int | None = None,
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
    statistics = _tracking_statistics(tracks, detections, num_frames=num_frames)
    files["metadata"].write_text(
        json.dumps(
            {
                **statistics,
                "tracker_backend": tracker_backend,
                "distance_metric": "squared Euclidean" if tracker_backend == "laptrack" else "Euclidean",
                "max_distance_pixels": max_distance,
                "max_distance_micrometres": max_distance * 0.65,
                "max_gap_frames": max_gap,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if first_frame is not None:
        files["overlay"] = output_path / "tracks_overlay.png"
        files["summary"] = output_path / "tracking_summary.png"
        save_track_overlay(first_frame, tracks, files["overlay"])
        save_tracking_summary(first_frame, tracks, detections, files["summary"], num_frames=num_frames)
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
    tracks, backend = _track_centroids_with_backend(detections, max_distance, max_gap, use_laptrack)
    first_frame = io.imread(first_frame_path) if first_frame_path is not None else None
    files = save_tracking_outputs(
        tracks,
        detections,
        output_dir=output_dir,
        first_frame=first_frame,
        tracker_backend=backend,
        max_distance=max_distance,
        max_gap=max_gap,
        num_frames=len(label_images),
    )
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


def save_tracking_summary(
    first_frame: np.ndarray,
    tracks: list[TrackPoint],
    detections: list[Detection],
    output_path: str | Path,
    num_frames: int | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].imshow(first_frame, cmap="gray")
    by_track: dict[int, list[TrackPoint]] = {}
    for point in tracks:
        by_track.setdefault(point.track_id, []).append(point)
    colours = plt.get_cmap("tab20")
    for index, points in enumerate(by_track.values()):
        ordered = sorted(points, key=lambda point: point.frame)
        xs = [point.x for point in ordered]
        ys = [point.y for point in ordered]
        colour = colours(index % 20)
        axes[0].plot(xs, ys, color=colour, linewidth=1.2)
        axes[0].scatter(xs[0], ys[0], color=[colour], marker="o", s=16)
        axes[0].scatter(xs[-1], ys[-1], color=[colour], marker="x", s=22)
    axes[0].set_title("Full trajectories on t000 (start: o, end: x)")
    axes[0].axis("off")

    frame_count = num_frames if num_frames is not None else (max((d.frame for d in detections), default=-1) + 1)
    counts = np.zeros(frame_count, dtype=int)
    for detection in detections:
        if 0 <= detection.frame < frame_count:
            counts[detection.frame] += 1
    axes[1].plot(np.arange(frame_count), counts, linewidth=1)
    axes[1].set(xlabel="Frame", ylabel="Detections", title="Detections per frame")
    axes[1].grid(True, alpha=0.3)

    lengths = [len(points) for points in by_track.values()]
    bins = np.arange(0.5, max(lengths, default=1) + 1.5, 1)
    axes[2].hist(lengths, bins=bins, color="tab:blue", edgecolor="white")
    axes[2].set(xlabel="Track length (frames)", ylabel="Tracks", title="Track fragmentation")
    axes[2].grid(True, axis="y", alpha=0.3)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _tracking_statistics(
    tracks: list[TrackPoint],
    detections: list[Detection],
    num_frames: int | None,
) -> dict[str, int | float]:
    by_track: dict[int, list[TrackPoint]] = {}
    for point in tracks:
        by_track.setdefault(point.track_id, []).append(point)
    lengths = [len(points) for points in by_track.values()]
    step_distances: list[float] = []
    for points in by_track.values():
        ordered = sorted(points, key=lambda point: point.frame)
        for previous, current in zip(ordered, ordered[1:]):
            frame_delta = current.frame - previous.frame
            if frame_delta > 0:
                step_distances.append(float(np.hypot(current.y - previous.y, current.x - previous.x) / frame_delta))
    frame_count = num_frames if num_frames is not None else (max((d.frame for d in detections), default=-1) + 1)
    detected_frames = len({detection.frame for detection in detections})
    return {
        "num_frames": frame_count,
        "num_detections": len(detections),
        "num_track_points": len(tracks),
        "num_tracks": len(by_track),
        "empty_frames": max(0, frame_count - detected_frames),
        "median_track_length": float(np.median(lengths)) if lengths else 0.0,
        "maximum_track_length": max(lengths, default=0),
        "tracks_at_most_three_frames": sum(length <= 3 for length in lengths),
        "median_step_pixels_per_frame": float(np.median(step_distances)) if step_distances else 0.0,
        "p95_step_pixels_per_frame": float(np.percentile(step_distances, 95)) if step_distances else 0.0,
        "maximum_step_pixels_per_frame": max(step_distances, default=0.0),
    }


def _write_csv(path: Path, rows: list[object], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})
