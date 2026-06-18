from __future__ import annotations

import numpy as np

from image_analysis_coursework.motion import (
    compute_derivatives,
    create_synthetic_translation,
    flow_energy,
    horn_schunck,
    local_average,
)


def test_compute_derivatives_shapes_and_temporal_difference():
    first = np.zeros((8, 8), dtype=np.float32)
    second = np.ones((8, 8), dtype=np.float32)

    fx, fy, ft = compute_derivatives(first, second)

    assert fx.shape == first.shape
    assert fy.shape == first.shape
    assert ft.shape == first.shape
    assert np.allclose(ft, 1.0)


def test_local_average_uses_four_neighbours():
    field = np.zeros((5, 5), dtype=np.float32)
    field[2, 2] = 4.0

    averaged = local_average(field)

    assert averaged[1, 2] == 1.0
    assert averaged[2, 1] == 1.0
    assert averaged[2, 2] == 0.0


def test_horn_schunck_energy_decreases_on_synthetic_pair():
    first, second = create_synthetic_translation(size=32, shift=(0, 1))

    result = horn_schunck(first, second, alpha=1.0, num_iterations=25)

    assert result.u.shape == first.shape
    assert result.v.shape == first.shape
    assert result.energy[-1] <= result.energy[0]


def test_flow_energy_is_zero_for_identical_images_and_zero_flow():
    image = np.zeros((8, 8), dtype=np.float32)
    fx, fy, ft = compute_derivatives(image, image)
    energy = flow_energy(np.zeros_like(image), np.zeros_like(image), fx, fy, ft, alpha=1.0)

    assert energy == 0.0
