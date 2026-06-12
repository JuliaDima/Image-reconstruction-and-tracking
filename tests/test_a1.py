from __future__ import annotations

import numpy as np
import skimage as ski

from image_analysis_coursework.a1 import load_ground_truth, simulate_acquisition


def test_load_ground_truth_normalizes_to_unit_interval(tmp_path):
    image = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    image_path = tmp_path / "sample.tif"
    ski.io.imsave(image_path, image)

    loaded = load_ground_truth(image_path)

    assert loaded.dtype == np.float32
    assert loaded.shape == image.shape
    assert np.min(loaded) == 0.0
    assert np.max(loaded) == 1.0


def test_simulate_acquisition_preserves_shape_and_range():
    image = np.zeros((16, 16), dtype=np.float32)
    image[4:12, 4:12] = 1.0

    result = simulate_acquisition(image, blur_sigma=2.0, noise_std=0.001, seed=42)

    assert result.ground_truth.shape == image.shape
    assert result.blurred.shape == image.shape
    assert result.noise.shape == image.shape
    assert result.blurred_noisy.shape == image.shape
    assert np.min(result.blurred) >= 0.0
    assert np.max(result.blurred) <= 1.0
    assert np.min(result.blurred_noisy) >= 0.0
    assert np.max(result.blurred_noisy) <= 1.0


def test_noise_is_deterministic_for_fixed_seed():
    image = np.full((8, 8), 0.5, dtype=np.float32)

    first = simulate_acquisition(image, seed=123)
    second = simulate_acquisition(image, seed=123)

    np.testing.assert_allclose(first.noise, second.noise)
    np.testing.assert_allclose(first.blurred_noisy, second.blurred_noisy)
