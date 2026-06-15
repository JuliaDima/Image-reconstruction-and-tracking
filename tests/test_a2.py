from __future__ import annotations

import numpy as np

from image_analysis_coursework.a1 import simulate_acquisition
from image_analysis_coursework.a2 import (
    image_gradient,
    run_classical_restoration,
    total_variation,
    tv_denoise_primal_dual,
)


def test_total_variation_is_zero_for_constant_image():
    image = np.ones((8, 8), dtype=np.float32) * 0.5

    assert total_variation(image) == 0.0


def test_image_gradient_preserves_shape():
    image = np.arange(16, dtype=np.float32).reshape(4, 4)

    grad_x, grad_y = image_gradient(image)

    assert grad_x.shape == image.shape
    assert grad_y.shape == image.shape


def test_tv_denoise_returns_objective_trace_and_valid_range():
    rng = np.random.default_rng(1)
    image = np.zeros((16, 16), dtype=np.float32)
    image[4:12, 4:12] = 1.0
    noisy = np.clip(image + rng.normal(0, 0.05, image.shape), 0.0, 1.0).astype(np.float32)

    restored, objective = tv_denoise_primal_dual(noisy, lambda_value=0.02, num_iterations=10)

    assert restored.shape == image.shape
    assert len(objective) == 11
    assert np.min(restored) >= 0.0
    assert np.max(restored) <= 1.0
    assert objective[-1] <= objective[0]


def test_run_classical_restoration_returns_one_result_per_lambda():
    image = np.zeros((24, 24), dtype=np.float32)
    image[6:18, 6:18] = 1.0
    acquisition = simulate_acquisition(image, noise_std=0.001, seed=2)

    result = run_classical_restoration(
        acquisition,
        lambda_values=(0.001, 0.01),
        num_iterations=5,
        gaussian_sigma=1.0,
    )

    assert len(result.tv_results) == 2
    assert result.gaussian_restored.shape == image.shape
    for tv_result in result.tv_results:
        assert tv_result.restored.shape == image.shape
        assert len(tv_result.objective) == 6
