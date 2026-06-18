from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import io

from image_analysis_coursework.yolo_segmentation import (
    export_dataset_to_yolo,
    label_image_to_yolo_lines,
    prediction_to_label_image,
    yolo_lines_to_label_image,
)


def test_label_image_round_trip_to_yolo_lines():
    labels = np.zeros((20, 20), dtype=np.uint16)
    labels[4:10, 5:12] = 1

    lines = label_image_to_yolo_lines(labels)
    restored = yolo_lines_to_label_image(lines, labels.shape)

    assert len(lines) == 1
    assert lines[0].startswith("0 ")
    assert restored.shape == labels.shape
    assert restored.max() == 1


def test_prediction_to_label_image_from_ultralytics_like_object():
    class Masks:
        xy = [np.array([[2, 2], [7, 2], [7, 7], [2, 7]], dtype=float)]

    class Prediction:
        masks = Masks()

    labels = prediction_to_label_image(Prediction(), (10, 10))

    assert labels.shape == (10, 10)
    assert labels.max() == 1
    assert labels[4, 4] == 1


def test_export_dataset_to_yolo_minimal_tree(tmp_path: Path):
    dataset = tmp_path / "PhC-C2DH-U373"
    for sequence in ("01", "02"):
        (dataset / sequence).mkdir(parents=True)
        (dataset / f"{sequence}_GT" / "SEG").mkdir(parents=True)
        image = np.zeros((16, 16), dtype=np.uint8)
        image[4:10, 5:11] = 255
        labels = np.zeros((16, 16), dtype=np.uint16)
        labels[4:10, 5:11] = 1
        io.imsave(dataset / sequence / "t000.tif", image)
        io.imsave(dataset / f"{sequence}_GT" / "SEG" / "man_seg000.tif", labels)

    records = export_dataset_to_yolo(dataset, output_dir=tmp_path / "yolo")

    assert len(records) == 2
    assert (tmp_path / "yolo" / "data.yaml").exists()
    assert records[0].label_path.read_text(encoding="utf-8").startswith("0 ")
