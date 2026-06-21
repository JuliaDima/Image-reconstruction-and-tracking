from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from skimage import io

from image_analysis_coursework.yolo_segmentation import (
    export_dataset_to_yolo,
    export_roundtrip_iou,
    label_image_to_yolo_lines,
    prediction_to_label_image,
    segment_image_with_model,
    train_yolo_model,
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
    assert export_roundtrip_iou(records[0]) > 0.9


def test_segment_image_with_model_propagates_confidence(tmp_path: Path):
    image_path = tmp_path / "image.tif"
    io.imsave(image_path, np.zeros((10, 12), dtype=np.uint8))

    class FakeModel:
        def __init__(self):
            self.kwargs = None

        def predict(self, image, **kwargs):
            self.kwargs = kwargs
            assert image.shape == (10, 12, 3)
            return [SimpleNamespace(masks=None)]

    model = FakeModel()
    labels = segment_image_with_model(model, image_path, confidence=0.4)

    assert labels.shape == (10, 12)
    assert model.kwargs == {"conf": 0.4, "verbose": False}


def test_train_yolo_model_writes_compact_metrics_to_requested_directory(tmp_path: Path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "data.yaml").write_text("names:\n  0: cell\n", encoding="utf-8")
    output_dir = tmp_path / "runs"

    class FakeYOLO:
        def __init__(self, model_name):
            self.model_name = model_name

        def train(self, **kwargs):
            assert Path(kwargs["project"]).is_absolute()
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            return SimpleNamespace(save_dir=save_dir)

        def val(self, **kwargs):
            return SimpleNamespace(
                results_dict={
                    "metrics/precision(M)": 0.8,
                    "metrics/recall(M)": 0.7,
                    "metrics/mAP50(M)": 0.75,
                    "metrics/mAP50-95(M)": 0.6,
                }
            )

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    summary = train_yolo_model(dataset_dir, output_dir=output_dir, epochs=2, imgsz=128, device="cpu")

    assert summary["map50_95_m"] == 0.6
    assert summary["best_checkpoint"].endswith("cells/weights/best.pt")
    metrics = (output_dir / "yolo_metrics.json").read_text(encoding="utf-8")
    assert "results_dict" not in metrics
    assert len(metrics) < 2_000
