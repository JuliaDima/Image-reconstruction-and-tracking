#!/usr/bin/env python
"""Generate supplementary analysis figures for the report.

Two cheap, CPU-only figures that reuse the Part I.A code paths:

* ``psf_mtf.png``   - the Gaussian point-spread function and its modulation
  transfer function, illustrating which spatial frequencies the forward
  operator attenuates and therefore which detail the inverse problem must
  recover.
* ``a2_error_maps.png`` - absolute reconstruction-error maps for the noisy
  observation, the Gaussian-filter baseline, and TV denoising at three
  regularisation strengths, making the edge-preservation argument concrete.

Neither figure needs a GPU or the trained models.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np

from image_analysis_coursework.a1 import (
    find_default_ground_truth,
    load_ground_truth,
    simulate_acquisition,
)
from image_analysis_coursework.a2 import run_classical_restoration


def gaussian_psf(kernel_size: int, sigma: float) -> np.ndarray:
    axis = np.arange(kernel_size) - kernel_size // 2
    xx, yy = np.meshgrid(axis, axis)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return psf / psf.sum()


def radial_average(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = np.array(image.shape) / 2.0 - 0.5
    yy, xx = np.indices(image.shape)
    radius = np.sqrt((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2)
    radius_int = radius.astype(int)
    totals = np.bincount(radius_int.ravel(), image.ravel())
    counts = np.bincount(radius_int.ravel())
    profile = totals / np.maximum(counts, 1)
    return np.arange(profile.size), profile


def save_psf_mtf(output_path: Path, sigma: float = 2.0, kernel_size: int = 11) -> None:
    psf = gaussian_psf(kernel_size, sigma)

    # Modulation transfer function from a finely sampled, zero-padded PSF.
    pad = 256
    big = gaussian_psf(pad, sigma)
    mtf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(big))))
    mtf /= mtf.max()
    freqs, mtf_radial = radial_average(mtf)
    freq_axis = freqs / pad  # cycles per pixel
    half_power = freq_axis[np.argmin(np.abs(mtf_radial - 0.5))]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    im = axes[0].imshow(psf, cmap="viridis")
    axes[0].set_title(f"Gaussian PSF, $\\sigma={sigma:g}$ ({kernel_size}$\\times${kernel_size})")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    centre_row = psf[kernel_size // 2]
    axes[1].plot(np.arange(kernel_size) - kernel_size // 2, centre_row, marker="o")
    axes[1].set(xlabel="Pixel offset", ylabel="Weight", title="PSF central profile")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(freq_axis[: pad // 2], mtf_radial[: pad // 2])
    axes[2].axhline(0.5, color="0.6", linestyle=":")
    axes[2].axvline(half_power, color="tab:red", linestyle="--", label=f"half-power: {half_power:.3f} cyc/px")
    axes[2].set(xlabel="Spatial frequency (cycles/pixel)", ylabel="MTF", title="Modulation transfer function")
    axes[2].set_xlim(0, 0.35)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"half-power cutoff = {half_power:.4f} cycles/pixel "
          f"(~{1.0 / half_power:.1f} px, ~{0.65 / half_power:.1f} micrometres)")


def save_error_maps(output_path: Path) -> None:
    ground_truth = load_ground_truth(find_default_ground_truth("data"))
    acquisition = simulate_acquisition(ground_truth, blur_sigma=2.0, noise_std=0.001, seed=42)
    result = run_classical_restoration(acquisition, num_iterations=100)

    gt = result.acquisition.ground_truth
    panels = [
        (f"Blurred+noise\nPSNR {result.noisy_psnr:.2f}", result.acquisition.blurred_noisy),
        (f"Gaussian $\\sigma=1$\nPSNR {result.gaussian_psnr:.2f}", result.gaussian_restored),
    ]
    panels.extend(
        (f"TV $\\lambda={tv.lambda_value:g}$\nPSNR {tv.psnr:.2f}", tv.restored)
        for tv in result.tv_results
    )

    n_panels = len(panels)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.6 * n_cols, 4.0 * n_rows), constrained_layout=True
    )
    flat_axes = np.atleast_1d(axes).ravel()
    last_im = None
    for axis, (title, image) in zip(flat_axes, panels, strict=False):
        last_im = axis.imshow(np.abs(gt - image), cmap="inferno", vmin=0.0, vmax=0.2)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    for axis in flat_axes[n_panels:]:
        axis.axis("off")
    fig.colorbar(last_im, ax=flat_axes.tolist(), fraction=0.03, pad=0.01, label="|error|")
    fig.suptitle("Absolute reconstruction error against the ground truth", fontsize=13)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    out_dir = REPO_ROOT / "outputs" / "extra"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_psf_mtf(out_dir / "psf_mtf.png")
    save_error_maps(out_dir / "a2_error_maps.png")
    print(f"wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
