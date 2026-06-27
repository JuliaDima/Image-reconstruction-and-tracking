from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from skimage import io

import image_analysis_coursework.tracking as tracking_module

from image_analysis_coursework.tracking import (
    extract_centroids,
    nearest_neighbour_tracks,
    save_tracking_outputs,
    segment_sequence_with_model,
    track_centroids,
)


def test_extract_centroids_from_label_image():
    labels = np.zeros((10, 10), dtype=np.uint16)
    labels[2:4, 3:5] = 1

    detections = extract_centroids(labels, frame_index=7)

    assert len(detections) == 1
    assert detections[0].frame == 7
    assert detections[0].label == 1
    assert detections[0].area == 4


def test_nearest_neighbour_tracks_links_simple_motion():
    frames = []
    for offset in range(3):
        labels = np.zeros((12, 12), dtype=np.uint16)
        labels[3:5, 3 + offset : 5 + offset] = 1
        frames.extend(extract_centroids(labels, frame_index=offset))

    tracks = nearest_neighbour_tracks(frames, max_distance=5.0)

    assert len(tracks) == 3
    assert len({point.track_id for point in tracks}) == 1


def test_track_centroids_fallback_runs_without_laptrack():
    labels = np.zeros((10, 10), dtype=np.uint16)
    labels[1:3, 1:3] = 1
    detections = extract_centroids(labels, frame_index=0)

    tracks = track_centroids(detections, use_laptrack=False)

    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_nearest_neighbour_tracks_closes_one_missing_frame():
    detections = []
    for frame, offset in ((0, 0), (2, 2)):
        labels = np.zeros((12, 12), dtype=np.uint16)
        labels[3:5, 3 + offset : 5 + offset] = 1
        detections.extend(extract_centroids(labels, frame_index=frame))

    tracks = nearest_neighbour_tracks(detections, max_distance=5.0, max_gap=1)

    assert len({point.track_id for point in tracks}) == 1


def test_laptrack_runtime_error_is_not_silently_hidden(monkeypatch):
    import unittest.mock as mock

    class BrokenLapTrack:
        def __init__(self, **kwargs):
            raise RuntimeError("invalid tracker configuration")

    fake_pd = mock.MagicMock()
    fake_pd.DataFrame.return_value.empty = False

    monkeypatch.setitem(sys.modules, "pandas", fake_pd)
    monkeypatch.setitem(sys.modules, "laptrack", SimpleNamespace(LapTrack=BrokenLapTrack))
    labels = np.zeros((10, 10), dtype=np.uint16)
    labels[1:3, 1:3] = 1

    with pytest.raises(RuntimeError, match="invalid tracker configuration"):
        track_centroids(extract_centroids(labels, frame_index=0), use_laptrack=True)


def test_tracking_metadata_records_backend_and_parameters(tmp_path: Path):
    labels = np.zeros((10, 10), dtype=np.uint16)
    labels[1:3, 1:3] = 1
    detections = extract_centroids(labels, frame_index=0)
    tracks = track_centroids(detections, use_laptrack=False)

    files = save_tracking_outputs(
        tracks,
        detections,
        output_dir=tmp_path,
        tracker_backend="nearest-neighbour",
        max_distance=35.0,
        max_gap=1,
        num_frames=2,
    )
    metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))

    assert metadata["tracker_backend"] == "nearest-neighbour"
    assert metadata["max_distance_pixels"] == 35.0
    assert metadata["max_distance_micrometres"] == pytest.approx(22.75)
    assert metadata["empty_frames"] == 1


def test_sequence_segmentation_loads_model_once(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    sequence_dir = dataset_root / "01"
    sequence_dir.mkdir(parents=True)
    for index in range(3):
        io.imsave(sequence_dir / f"t{index:03d}.tif", np.zeros((8, 8), dtype=np.uint8))

    calls = {"loads": 0, "predictions": 0}
    model = object()

    def fake_load(path):
        calls["loads"] += 1
        return model

    def fake_segment(loaded_model, image_path, confidence):
        assert loaded_model is model
        assert confidence == 0.4
        calls["predictions"] += 1
        return np.zeros((8, 8), dtype=np.uint16)

    monkeypatch.setattr(tracking_module, "download_dataset", lambda data_dir: dataset_root)
    monkeypatch.setattr(tracking_module, "load_yolo_model", fake_load)
    monkeypatch.setattr(tracking_module, "segment_image_with_model", fake_segment)

    paths = segment_sequence_with_model("model.pt", data_dir=tmp_path, output_dir=tmp_path / "labels", confidence=0.4)

    assert len(paths) == 3
    assert calls == {"loads": 1, "predictions": 3}
