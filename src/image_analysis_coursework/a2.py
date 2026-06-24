"""Part I.A.2 classical image restoration with total variation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from image_analysis_coursework.a1 import (
    AcquisitionResult,
    find_default_ground_truth,
    load_ground_truth,
    save_a1_outputs,
    simulate_acquisition,
)


@dataclass(frozen=True)
class TVDenoisingResult:
    """Result for one TV regularisation setting."""

    lambda_value: float
    restored: np.ndarray
    objective: list[float]
    psnr: float
    ssim: float


@dataclass(frozen=True)
class ClassicalRestorationResult:
    """Outputs for Part I.A.2."""

    acquisition: AcquisitionResult
    tv_results: list[TVDenoisingResult]
    gaussian_restored: np.ndarray
    gaussian_sigma: float
    gaussian_psnr: float
    gaussian_ssim: float
    noisy_psnr: float
    noisy_ssim: float


def image_gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward finite differences with Neumann boundary conditions."""

    grad_x = np.zeros_like(image, dtype=np.float32)
    grad_y = np.zeros_like(image, dtype=np.float32)
    grad_x[:, :-1] = image[:, 1:] - image[:, :-1]
    grad_y[:-1, :] = image[1:, :] - image[:-1, :]
    return grad_x, grad_y


def divergence(field_x: np.ndarray, field_y: np.ndarray) -> np.ndarray:
    """Adjoint of the forward-difference gradient."""

    div = np.zeros_like(field_x, dtype=np.float32)
    div[:, :-1] -= field_x[:, :-1]
    div[:, 1:] += field_x[:, :-1]
    div[:-1, :] -= field_y[:-1, :]
    div[1:, :] += field_y[:-1, :]
    return div


def total_variation(image: np.ndarray) -> float:
    """Isotropic discrete total variation."""

    grad_x, grad_y = image_gradient(image)
    return float(np.sum(np.sqrt(grad_x**2 + grad_y**2)))


def tv_objective(restored: np.ndarray, observation: np.ndarray, lambda_value: float) -> float:
    """Objective value: 0.5 ||g - f||_2^2 + lambda TV(g)."""

    residual = restored - observation
    data_term = 0.5 * float(np.sum(residual**2))
    return data_term + lambda_value * total_variation(restored)


def tv_denoise_primal_dual(
    observation: np.ndarray,
    lambda_value: float,
    num_iterations: int = 250,
    tau: float = 0.25,
    sigma: float = 0.25,
    theta: float = 1.0,
) -> tuple[np.ndarray, list[float]]:
    """Approximately solve ROF TV denoising using Chambolle-Pock iterations."""

    if lambda_value <= 0:
        raise ValueError("lambda_value must be positive")
    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")

    f = np.asarray(observation, dtype=np.float32)
    x = f.copy()
    x_bar = x.copy()
    dual_x = np.zeros_like(f, dtype=np.float32)
    dual_y = np.zeros_like(f, dtype=np.float32)
    objective = [tv_objective(x, f, lambda_value)]

    for _ in range(num_iterations):
        grad_x, grad_y = image_gradient(x_bar)
        dual_x += sigma * grad_x
        dual_y += sigma * grad_y
        norm = np.maximum(1.0, np.sqrt(dual_x**2 + dual_y**2) / lambda_value)
        dual_x /= norm
        dual_y /= norm

        previous_x = x
        x = (x - tau * divergence(dual_x, dual_y) + tau * f) / (1.0 + tau)
        x = np.clip(x, 0.0, 1.0).astype(np.float32)
        x_bar = x + theta * (x - previous_x)
        objective.append(tv_objective(x, f, lambda_value))

    return x, objective


def evaluate_against_ground_truth(ground_truth: np.ndarray, image: np.ndarray) -> tuple[float, float]:
    """Return PSNR and SSIM for a restored image."""

    return (
        float(peak_signal_noise_ratio(ground_truth, image, data_range=1.0)),
        float(structural_similarity(ground_truth, image, data_range=1.0)),
    )


def run_classical_restoration(
    acquisition: AcquisitionResult,
    lambda_values: tuple[float, ...] = (0.001, 0.01, 0.08),
    num_iterations: int = 250,
    gaussian_sigma: float = 1.0,
) -> ClassicalRestorationResult:
    """Run TV denoising for several lambdas and a Gaussian-filter baseline."""

    tv_results: list[TVDenoisingResult] = []
    for lambda_value in lambda_values:
        restored, objective = tv_denoise_primal_dual(
            acquisition.blurred_noisy,
            lambda_value=lambda_value,
            num_iterations=num_iterations,
        )
        restored_psnr, restored_ssim = evaluate_against_ground_truth(
            acquisition.ground_truth,
            restored,
        )
        tv_results.append(
            TVDenoisingResult(
                lambda_value=lambda_value,
                restored=restored,
                objective=objective,
                psnr=restored_psnr,
                ssim=restored_ssim,
            )
        )

    gaussian_restored = ski.filters.gaussian(
        acquisition.blurred_noisy,
        sigma=gaussian_sigma,
        preserve_range=True,
    ).astype(np.float32)
    gaussian_restored = np.clip(gaussian_restored, 0.0, 1.0)
    gaussian_psnr, gaussian_ssim = evaluate_against_ground_truth(
        acquisition.ground_truth,
        gaussian_restored,
    )
    noisy_psnr, noisy_ssim = evaluate_against_ground_truth(
        acquisition.ground_truth,
        acquisition.blurred_noisy,
    )

    return ClassicalRestorationResult(
        acquisition=acquisition,
        tv_results=tv_results,
        gaussian_restored=gaussian_restored,
        gaussian_sigma=gaussian_sigma,
        gaussian_psnr=gaussian_psnr,
        gaussian_ssim=gaussian_ssim,
        noisy_psnr=noisy_psnr,
        noisy_ssim=noisy_ssim,
    )


def save_a2_outputs(
    result: ClassicalRestorationResult,
    output_dir: str | Path = "outputs/part_i/a2",
) -> dict[str, Path]:
    """Save A.2 restoration images, objective traces, and summary metadata."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {
        "comparison": output_path / "comparison.png",
        "objective": output_path / "objective.png",
        "metadata": output_path / "metadata.json",
        "gaussian": output_path / "gaussian.png",
    }

    ski.io.imsave(files["gaussian"], ski.util.img_as_ubyte(result.gaussian_restored))
    for tv_result in result.tv_results:
        key = f"tv_lambda_{tv_result.lambda_value:g}"
        path = output_path / f"{key}.png"
        files[key] = path
        ski.io.imsave(path, ski.util.img_as_ubyte(tv_result.restored))

    _save_restoration_comparison(result, files["comparison"])
    _save_objective_plot(result.tv_results, files["objective"])

    metadata: dict[str, Any] = {
        "source_image": str(result.acquisition.source_image) if result.acquisition.source_image else None,
        "blur_sigma": result.acquisition.blur_sigma,
        "noise_std": result.acquisition.noise_std,
        "seed": result.acquisition.seed,
        "noisy_psnr": result.noisy_psnr,
        "noisy_ssim": result.noisy_ssim,
        "gaussian_sigma": result.gaussian_sigma,
        "gaussian_psnr": result.gaussian_psnr,
        "gaussian_ssim": result.gaussian_ssim,
        "tv_results": [
            {
                "lambda": tv_result.lambda_value,
                "psnr": tv_result.psnr,
                "ssim": tv_result.ssim,
                "initial_objective": tv_result.objective[0],
                "final_objective": tv_result.objective[-1],
                "num_iterations": len(tv_result.objective) - 1,
            }
            for tv_result in result.tv_results
        ],
    }
    files["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return files


def run_a2_from_image(
    image_path: str | Path | None = None,
    data_dir: str | Path = "data",
    a1_output_dir: str | Path = "outputs/part_i/a1",
    a2_output_dir: str | Path = "outputs/part_i/a2",
    blur_sigma: float = 2.0,
    noise_std: float = 0.001,
    seed: int = 42,
    lambda_values: tuple[float, ...] = (0.001, 0.01, 0.08),
    num_iterations: int = 250,
    gaussian_sigma: float = 1.0,
) -> tuple[ClassicalRestorationResult, dict[str, Path]]:
    """Load/generate A.1 data, run A.2, and save all relevant outputs."""

    source_image = Path(image_path) if image_path is not None else find_default_ground_truth(data_dir)
    ground_truth = load_ground_truth(source_image)
    acquisition = simulate_acquisition(
        ground_truth,
        blur_sigma=blur_sigma,
        noise_std=noise_std,
        seed=seed,
        source_image=source_image,
    )
    save_a1_outputs(acquisition, a1_output_dir)
    result = run_classical_restoration(
        acquisition,
        lambda_values=lambda_values,
        num_iterations=num_iterations,
        gaussian_sigma=gaussian_sigma,
    )
    files = save_a2_outputs(result, a2_output_dir)
    return result, files


def _save_restoration_comparison(result: ClassicalRestorationResult, output_path: Path) -> None:
    panels = [
        ("Ground truth", result.acquisition.ground_truth),
        (f"Blurred + noise\nPSNR {result.noisy_psnr:.2f}", result.acquisition.blurred_noisy),
        (f"Gaussian\nPSNR {result.gaussian_psnr:.2f}", result.gaussian_restored),
    ]
    panels.extend(
        (f"TV $\\lambda={tv_result.lambda_value:g}$\nPSNR {tv_result.psnr:.2f}", tv_result.restored)
        for tv_result in result.tv_results
    )
    _save_image_grid(panels, output_path)


def _save_image_grid(panels: list[tuple[str, np.ndarray]], output_path: Path, max_cols: int = 3) -> None:
    """Lay panels out on a grid with at most ``max_cols`` columns per row."""

    n_panels = len(panels)
    n_cols = min(max_cols, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.6 * n_cols, 4.0 * n_rows), constrained_layout=True
    )
    flat_axes = np.atleast_1d(axes).ravel()
    for axis, (title, image) in zip(flat_axes, panels, strict=False):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    for axis in flat_axes[n_panels:]:
        axis.axis("off")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_objective_plot(tv_results: list[TVDenoisingResult], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for tv_result in tv_results:
        axis.plot(tv_result.objective, label=f"lambda={tv_result.lambda_value:g}")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Objective")
    axis.set_title("TV objective during primal-dual iterations")
    axis.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
