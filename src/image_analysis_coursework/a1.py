"""Part I.A.1 forward image acquisition simulation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import zipfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pooch
import skimage as ski

DATASET_URL = "https://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip"
DATASET_HASH = "b18185c18fce54e8eeb93e4bbb9b201d757add9409bbf2283b8114185a11bc9e"
DEFAULT_SEQUENCE = "01"


@dataclass(frozen=True)
class AcquisitionResult:
    """Images and metadata produced by the A.1 acquisition model."""

    ground_truth: np.ndarray
    blurred: np.ndarray
    noise: np.ndarray
    blurred_noisy: np.ndarray
    blur_sigma: float
    noise_std: float
    seed: int
    source_image: Path | None = None


def download_dataset(data_dir: str | Path = "data") -> Path:
    """Download and extract the coursework microscopy dataset if needed."""

    data_root = Path(data_dir)
    dataset_root = data_root / "PhC-C2DH-U373"
    if dataset_root.exists():
        return dataset_root

    data_root.mkdir(parents=True, exist_ok=True)
    archive_path = pooch.retrieve(
        url=DATASET_URL,
        known_hash=f"sha256:{DATASET_HASH}",
        path=data_root / "downloads",
        progressbar=True,
    )
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(data_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset was not extracted to {dataset_root}")
    return dataset_root


def find_default_ground_truth(data_dir: str | Path = "data") -> Path:
    """Return the first sorted TIFF frame from the default training sequence."""

    dataset_root = download_dataset(data_dir)
    sequence_dir = dataset_root / DEFAULT_SEQUENCE
    candidates = sorted(sequence_dir.glob("*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No TIFF images found in {sequence_dir}")
    return candidates[0]


def load_ground_truth(path: str | Path) -> np.ndarray:
    """Load a TIFF image and normalize it to floating point values in [0, 1]."""

    image_path = Path(path)
    image = ski.io.imread(image_path)
    image = np.asarray(image, dtype=np.float32)

    if image.ndim > 2:
        image = image[..., 0]

    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image - min_value) / (max_value - min_value)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def simulate_acquisition(
    image: np.ndarray,
    blur_sigma: float = 2.0,
    noise_std: float = 0.001,
    seed: int = 42,
    source_image: str | Path | None = None,
) -> AcquisitionResult:
    """Simulate diffraction blur followed by additive Gaussian noise."""

    ground_truth = np.asarray(image, dtype=np.float32)
    if ground_truth.ndim != 2:
        raise ValueError("A.1 expects a single-channel 2D image")
    if blur_sigma <= 0:
        raise ValueError("blur_sigma must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be nonnegative")

    ground_truth = np.clip(ground_truth, 0.0, 1.0)
    blurred = ski.filters.gaussian(
        ground_truth,
        sigma=blur_sigma,
        preserve_range=True,
    ).astype(np.float32)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=ground_truth.shape).astype(np.float32)
    blurred_noisy = np.clip(blurred + noise, 0.0, 1.0).astype(np.float32)

    return AcquisitionResult(
        ground_truth=ground_truth,
        blurred=blurred,
        noise=noise,
        blurred_noisy=blurred_noisy,
        blur_sigma=blur_sigma,
        noise_std=noise_std,
        seed=seed,
        source_image=Path(source_image) if source_image is not None else None,
    )


def save_a1_outputs(
    result: AcquisitionResult,
    output_dir: str | Path = "outputs/part_i/a1",
) -> dict[str, Path]:
    """Save A.1 images, comparison figure, and metadata."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = {
        "ground_truth": output_path / "ground_truth.png",
        "blurred": output_path / "blurred.png",
        "blurred_noisy": output_path / "blurred_noisy.png",
        "comparison": output_path / "comparison.png",
        "metadata": output_path / "metadata.json",
    }

    ski.io.imsave(files["ground_truth"], ski.util.img_as_ubyte(result.ground_truth))
    ski.io.imsave(files["blurred"], ski.util.img_as_ubyte(np.clip(result.blurred, 0.0, 1.0)))
    ski.io.imsave(
        files["blurred_noisy"],
        ski.util.img_as_ubyte(result.blurred_noisy),
    )
    _save_comparison(result, files["comparison"])

    metadata: dict[str, Any] = {
        "source_image": str(result.source_image) if result.source_image else None,
        "blur_sigma": result.blur_sigma,
        "noise_std": result.noise_std,
        "seed": result.seed,
        "shape": list(result.ground_truth.shape),
        "ground_truth_min": float(np.min(result.ground_truth)),
        "ground_truth_max": float(np.max(result.ground_truth)),
        "blurred_min": float(np.min(result.blurred)),
        "blurred_max": float(np.max(result.blurred)),
        "blurred_noisy_min": float(np.min(result.blurred_noisy)),
        "blurred_noisy_max": float(np.max(result.blurred_noisy)),
    }
    files["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return files


def _save_comparison(result: AcquisitionResult, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [
        ("Ground truth", result.ground_truth),
        (f"Gaussian blur\n$\\sigma={result.blur_sigma:g}$", result.blurred),
        (f"Blurred + noise\n$\\sigma_n={result.noise_std:g}$", result.blurred_noisy),
    ]

    for axis, (title, image) in zip(axes, panels, strict=True):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
