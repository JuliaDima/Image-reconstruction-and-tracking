from __future__ import annotations

import numpy as np

from image_analysis_coursework.tracking import extract_centroids, nearest_neighbour_tracks, track_centroids


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
