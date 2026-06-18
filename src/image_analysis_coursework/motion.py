"""Part II.A Horn-Schunck optical-flow utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, exposure, io, transform, util

from image_analysis_coursework.a1 import download_dataset, load_ground_truth


@dataclass(frozen=True)
class OpticalFlowResult:
    u: np.ndarray
    v: np.ndarray
    energy: list[float]
    fx: np.ndarray
    fy: np.ndarray
    ft: np.ndarray
    alpha: float
    num_iterations: int


def load_image_pair(data_dir: str | Path = "data", sequence: str = "01", index: int = 0) -> tuple[np.ndarray, np.ndarray, tuple[Path, Path]]:
    dataset_root = download_dataset(data_dir)
    frames = sorted((dataset_root / sequence).glob("*.tif"))
    if index < 0 or index + 1 >= len(frames):
        raise IndexError(f"Need a consecutive pair at index {index}; found {len(frames)} frames")
    first = load_ground_truth(frames[index])
    second = load_ground_truth(frames[index + 1])
    return first, second, (frames[index], frames[index + 1])


def compute_derivatives(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError("Images must have matching shapes")
    mean_image = 0.5 * (first + second)
    fy, fx = np.gradient(mean_image)
    ft = second - first
    return fx.astype(np.float32), fy.astype(np.float32), ft.astype(np.float32)


def local_average(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float32)
    padded = np.pad(field, 1, mode="edge")
    return (
        padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
    ) / 4.0


def flow_energy(u: np.ndarray, v: np.ndarray, fx: np.ndarray, fy: np.ndarray, ft: np.ndarray, alpha: float) -> float:
    residual = fx * u + fy * v + ft
    ux, uy = np.gradient(u)
    vx, vy = np.gradient(v)
    smoothness = ux**2 + uy**2 + vx**2 + vy**2
    return float(np.mean(residual**2) + alpha * np.mean(smoothness))


def horn_schunck(first: np.ndarray, second: np.ndarray, alpha: float = 1.0, num_iterations: int = 200) -> OpticalFlowResult:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")
    fx, fy, ft = compute_derivatives(first, second)
    u = np.zeros_like(fx, dtype=np.float32)
    v = np.zeros_like(fx, dtype=np.float32)
    denom = alpha**2 + fx**2 + fy**2
    energy = [flow_energy(u, v, fx, fy, ft, alpha)]
    for _ in range(num_iterations):
        u_bar = local_average(u)
        v_bar = local_average(v)
        residual = fx * u_bar + fy * v_bar + ft
        u = u_bar - fx * residual / denom
        v = v_bar - fy * residual / denom
        energy.append(flow_energy(u, v, fx, fy, ft, alpha))
    return OpticalFlowResult(u=u, v=v, energy=energy, fx=fx, fy=fy, ft=ft, alpha=alpha, num_iterations=num_iterations)


def difference_image(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    diff = np.abs(np.asarray(second, dtype=np.float32) - np.asarray(first, dtype=np.float32))
    if float(diff.max()) > 0:
        diff = exposure.rescale_intensity(diff, in_range="image", out_range=(0.0, 1.0))
    return diff.astype(np.float32)


def flow_to_hsv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)
    hue = (angle + np.pi) / (2 * np.pi)
    value = magnitude / (float(magnitude.max()) + 1e-8)
    saturation = np.ones_like(value)
    hsv = np.stack([hue, saturation, value], axis=-1)
    return color.hsv2rgb(hsv)


def create_synthetic_translation(size: int = 64, shift: tuple[int, int] = (1, 0)) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((size, size), dtype=np.float32)
    image[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1.0
    image = transform.resize(image, (size, size), anti_aliasing=True).astype(np.float32)
    moved = np.roll(image, shift=shift, axis=(0, 1))
    return image, moved.astype(np.float32)


def save_optical_flow_outputs(
    first: np.ndarray,
    second: np.ndarray,
    result: OpticalFlowResult,
    output_dir: str | Path = "outputs/part_ii/a",
    frame_paths: tuple[Path, Path] | None = None,
    stride: int = 16,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    files = {
        "pair": output_path / "image_pair.png",
        "difference": output_path / "difference.png",
        "energy": output_path / "energy.png",
        "quiver": output_path / "flow_quiver.png",
        "hsv": output_path / "flow_hsv.png",
        "metadata": output_path / "metadata.json",
    }
    _save_pair(first, second, files["pair"])
    io.imsave(files["difference"], util.img_as_ubyte(difference_image(first, second)))
    _save_energy(result, files["energy"])
    _save_quiver(first, result, files["quiver"], stride=stride)
    io.imsave(files["hsv"], util.img_as_ubyte(np.clip(flow_to_hsv(result.u, result.v), 0.0, 1.0)))
    files["metadata"].write_text(
        json.dumps(
            {
                "frame_paths": [str(path) for path in frame_paths] if frame_paths else None,
                "alpha": result.alpha,
                "num_iterations": result.num_iterations,
                "initial_energy": result.energy[0],
                "final_energy": result.energy[-1],
                "mean_u": float(np.mean(result.u)),
                "mean_v": float(np.mean(result.v)),
                "mean_magnitude": float(np.mean(np.sqrt(result.u**2 + result.v**2))),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return files


def run_motion_experiment(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs/part_ii/a",
    sequence: str = "01",
    index: int = 0,
    alpha: float = 1.0,
    num_iterations: int = 200,
    stride: int = 16,
) -> tuple[OpticalFlowResult, dict[str, Path]]:
    first, second, frame_paths = load_image_pair(data_dir, sequence=sequence, index=index)
    result = horn_schunck(first, second, alpha=alpha, num_iterations=num_iterations)
    files = save_optical_flow_outputs(first, second, result, output_dir=output_dir, frame_paths=frame_paths, stride=stride)
    return result, files


def _save_pair(first: np.ndarray, second: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for axis, image, title in zip(axes, [first, second], ["Frame t", "Frame t+1"], strict=True):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_energy(result: OpticalFlowResult, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.plot(result.energy)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Energy")
    axis.set_title("Horn-Schunck energy")
    axis.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_quiver(first: np.ndarray, result: OpticalFlowResult, output_path: Path, stride: int) -> None:
    yy, xx = np.mgrid[0:first.shape[0]:stride, 0:first.shape[1]:stride]
    fig, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.imshow(first, cmap="gray", vmin=0.0, vmax=1.0)
    axis.quiver(xx, yy, result.u[::stride, ::stride], result.v[::stride, ::stride], color="tab:red", angles="xy")
    axis.set_title("Estimated optical flow")
    axis.axis("off")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
