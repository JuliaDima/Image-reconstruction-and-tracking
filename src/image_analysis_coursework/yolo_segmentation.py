"""Part II.B YOLO segmentation export, training, and prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import numpy as np
from skimage import draw, io, measure, util

from image_analysis_coursework.a1 import download_dataset, load_ground_truth


@dataclass(frozen=True)
class ExportRecord:
    image_path: Path
    label_path: Path
    source_image: Path
    source_label: Path
    split: str


def label_image_to_yolo_lines(labels: np.ndarray, class_index: int = 0, tolerance: float = 0.5) -> list[str]:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D label image")
    height, width = labels.shape
    lines: list[str] = []
    for prop in measure.regionprops(labels.astype(np.int32)):
        if prop.area < 3:
            continue
        mask = np.pad(prop.image_filled.astype(np.uint8), 2)
        contours = measure.find_contours(mask, level=0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        contour = measure.approximate_polygon(contour, tolerance)
        min_y, min_x, _, _ = prop.bbox
        coords: list[str] = []
        for y, x in contour:
            global_x = float(np.clip((x - 2 + min_x) / width, 0.0, 1.0))
            global_y = float(np.clip((y - 2 + min_y) / height, 0.0, 1.0))
            coords.extend([f"{global_x:.6f}", f"{global_y:.6f}"])
        if len(coords) >= 6:
            lines.append(f"{class_index} {' '.join(coords)}")
    return lines


def yolo_lines_to_label_image(lines: list[str], image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    label_image = np.zeros((height, width), dtype=np.uint16)
    next_label = 1
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            continue
        coords = np.asarray(parts[1:], dtype=float).reshape(-1, 2)
        xs = coords[:, 0] * width
        ys = coords[:, 1] * height
        rr, cc = draw.polygon(ys, xs, shape=label_image.shape)
        if rr.size:
            label_image[rr, cc] = next_label
            next_label += 1
    return label_image


def _image_for_segmentation_label(label_path: Path) -> Path:
    sequence = label_path.parents[1].name.replace("_GT", "")
    frame_name = label_path.name.replace("man_seg", "t")
    return label_path.parents[2] / sequence / frame_name


def export_dataset_to_yolo(
    dataset_root: str | Path,
    output_dir: str | Path = "outputs/part_ii/b/yolo_dataset",
    train_sequences: tuple[str, ...] = ("01",),
    val_sequences: tuple[str, ...] = ("02",),
) -> list[ExportRecord]:
    dataset_path = Path(dataset_root)
    output_path = Path(output_dir)
    records: list[ExportRecord] = []
    for split, sequences in (("train", train_sequences), ("val", val_sequences)):
        for sequence in sequences:
            seg_dir = dataset_path / f"{sequence}_GT" / "SEG"
            for label_path in sorted(seg_dir.glob("*.tif")):
                source_image = _image_for_segmentation_label(label_path)
                index = len(records)
                image_out = output_path / split / "images" / f"{sequence}_{index:06d}.png"
                label_out = output_path / split / "labels" / f"{sequence}_{index:06d}.txt"
                image_out.parent.mkdir(parents=True, exist_ok=True)
                label_out.parent.mkdir(parents=True, exist_ok=True)
                image = load_ground_truth(source_image)
                labels = io.imread(label_path)
                io.imsave(image_out, util.img_as_ubyte(image))
                label_out.write_text("\n".join(label_image_to_yolo_lines(labels)) + "\n", encoding="utf-8")
                records.append(ExportRecord(image_out, label_out, source_image, label_path, split))
    (output_path / "data.yaml").write_text(
        "train: ./train/images\nval: ./val/images\nnames:\n  0: cell\n",
        encoding="utf-8",
    )
    (output_path / "export_manifest.json").write_text(
        json.dumps([_record_to_json(record) for record in records], indent=2) + "\n",
        encoding="utf-8",
    )
    return records


def export_phc_dataset_to_yolo(data_dir: str | Path = "data", output_dir: str | Path = "outputs/part_ii/b/yolo_dataset") -> list[ExportRecord]:
    dataset_root = download_dataset(data_dir)
    return export_dataset_to_yolo(dataset_root, output_dir=output_dir)


def verify_yolo_export(record: ExportRecord, output_path: str | Path) -> Path:
    image = io.imread(record.image_path)
    lines = record.label_path.read_text(encoding="utf-8").splitlines()
    labels = yolo_lines_to_label_image(lines, image.shape[:2])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_overlay(image, labels, output, title=f"{record.split}: {record.source_label.name}")
    return output


def export_roundtrip_iou(record: ExportRecord) -> float:
    """Measure binary-mask agreement after YOLO polygon rasterisation."""

    original = io.imread(record.source_label) > 0
    lines = record.label_path.read_text(encoding="utf-8").splitlines()
    restored = yolo_lines_to_label_image(lines, original.shape) > 0
    union = np.logical_or(original, restored).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(original, restored).sum() / union)


def prediction_to_label_image(prediction: Any, image_shape: tuple[int, int]) -> np.ndarray:
    label_image = np.zeros(image_shape, dtype=np.uint16)
    masks = getattr(prediction, "masks", None)
    polygons = getattr(masks, "xy", None) if masks is not None else None
    if polygons is None:
        return label_image
    for index, polygon in enumerate(polygons, start=1):
        coords = np.asarray(polygon, dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 3:
            continue
        rr, cc = draw.polygon(coords[:, 1], coords[:, 0], shape=image_shape)
        label_image[rr, cc] = index
    return label_image


def train_yolo_model(
    yolo_dataset_dir: str | Path,
    output_dir: str | Path = "outputs/part_ii/b/yolo_runs",
    model_name: str = "yolo11n-seg.pt",
    epochs: int = 100,
    imgsz: int = 640,
    device: str | int | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    dataset_yaml = (Path(yolo_dataset_dir) / "data.yaml").resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)
    training_results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=str(output_path),
        name="cells",
        device=device,
        exist_ok=True,
    )
    metrics = model.val(data=str(dataset_yaml), imgsz=imgsz, device=device)
    values = getattr(metrics, "results_dict", {})
    save_dir = Path(getattr(training_results, "save_dir", output_path / "cells")).resolve()
    split_counts = _manifest_split_counts(Path(yolo_dataset_dir) / "export_manifest.json")
    summary = {
        "model": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "device": str(device) if device is not None else "auto",
        "dataset_yaml": str(dataset_yaml),
        "run_dir": str(save_dir),
        "best_checkpoint": str(save_dir / "weights" / "best.pt"),
        "train_frames": split_counts.get("train", 0),
        "validation_frames": split_counts.get("val", 0),
        "precision_m": _optional_float(values.get("metrics/precision(M)")),
        "recall_m": _optional_float(values.get("metrics/recall(M)")),
        "map50_m": _optional_float(values.get("metrics/mAP50(M)")),
        "map50_95_m": _optional_float(values.get("metrics/mAP50-95(M)")),
    }
    (output_path / "yolo_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def load_yolo_model(model_path: str | Path) -> Any:
    """Load an Ultralytics model once for one or more predictions."""

    from ultralytics import YOLO

    return YOLO(str(model_path))


def segment_image_with_model(
    model: Any,
    image_path: str | Path,
    image_shape: tuple[int, int] | None = None,
    confidence: float = 0.25,
) -> np.ndarray:
    """Segment one image with an already-loaded Ultralytics model."""

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must lie in [0, 1]")
    image = io.imread(image_path)
    if image_shape is None:
        image_shape = image.shape[:2]
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    predictions = model.predict(image, conf=confidence, verbose=False)
    return prediction_to_label_image(predictions[0], image_shape)


def segment_image(
    model_path: str | Path,
    image_path: str | Path,
    image_shape: tuple[int, int] | None = None,
    confidence: float = 0.25,
) -> np.ndarray:
    """Load a model and segment one image."""

    model = load_yolo_model(model_path)
    return segment_image_with_model(model, image_path, image_shape=image_shape, confidence=confidence)


def overlay_prediction(
    model_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    confidence: float = 0.25,
) -> Path:
    image = io.imread(image_path)
    labels = segment_image(model_path, image_path, image_shape=image.shape[:2], confidence=confidence)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_overlay(image, labels, output, title=Path(image_path).name)
    return output


def save_prediction_montage(
    model_path: str | Path,
    image_paths: list[str | Path],
    output_path: str | Path,
    confidence: float = 0.25,
) -> Path:
    """Overlay predictions on several frames using one loaded model."""

    if not image_paths:
        raise ValueError("image_paths must not be empty")
    model = load_yolo_model(model_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(image_paths), figsize=(5 * len(image_paths), 4.2), constrained_layout=True)
    for axis, image_path in zip(np.atleast_1d(axes), image_paths, strict=True):
        image = io.imread(image_path)
        labels = segment_image_with_model(model, image_path, image_shape=image.shape[:2], confidence=confidence)
        axis.imshow(image, cmap="gray")
        axis.imshow(np.ma.masked_where(labels == 0, labels), cmap="tab20", alpha=0.45)
        axis.set_title(f"{Path(image_path).name}: {int(labels.max())} masks")
        axis.axis("off")
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _save_overlay(image: np.ndarray, labels: np.ndarray, output_path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.imshow(image, cmap="gray")
    masked = np.ma.masked_where(labels == 0, labels)
    axis.imshow(masked, cmap="tab20", alpha=0.45)
    axis.set_title(title)
    axis.axis("off")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _record_to_json(record: ExportRecord) -> dict[str, str]:
    return {
        "image_path": str(record.image_path),
        "label_path": str(record.label_path),
        "source_image": str(record.source_image),
        "source_label": str(record.source_label),
        "split": record.split,
    }


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _manifest_split_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for record in records:
        split = str(record["split"])
        counts[split] = counts.get(split, 0) + 1
    return counts
