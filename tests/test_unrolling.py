from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from image_analysis_coursework.unrolling import (
    GaussianBlurOperator,
    SharedUnrolledPGD,
    UnrolledPGD,
    make_blurry_observation,
    train_unrolled_pgd,
)


def test_gaussian_blur_operator_preserves_shape():
    operator = GaussianBlurOperator(kernel_size=11, sigma=2.0)
    image = torch.rand(2, 1, 16, 16)

    blurred = operator.A(image)
    adjoint = operator.AT(blurred)

    assert blurred.shape == image.shape
    assert adjoint.shape == image.shape


def test_unrolled_pgd_forward_shape_and_step_positivity():
    operator = GaussianBlurOperator(kernel_size=11, sigma=2.0)
    model = UnrolledPGD(operator=operator, n_iter=2, features=4)
    image = torch.rand(1, 1, 16, 16)
    blurry = make_blurry_observation(operator, image, noise_var=0.001)

    output = model(blurry)

    assert output.shape == image.shape
    assert torch.min(output) >= 0.0
    assert torch.max(output) <= 1.0
    assert all(step > 0 for step in model.positive_steps())


def test_train_unrolled_pgd_smoke():
    operator = GaussianBlurOperator(kernel_size=11, sigma=2.0)
    model = UnrolledPGD(operator=operator, n_iter=1, features=4)
    class TinyDataset(Dataset):
        def __init__(self) -> None:
            self.images = torch.rand(2, 1, 16, 16)

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, index: int) -> torch.Tensor:
            return self.images[index]

    loader = DataLoader(TinyDataset(), batch_size=1)
    history = train_unrolled_pgd(
        model,
        loader,
        operator,
        device=torch.device("cpu"),
        epochs=1,
        noise_var=0.001,
        learning_rate=1e-4,
    )

    assert len(history.epochs) == 1
    assert history.epochs[0]["loss"] >= 0.0


def test_shared_unrolled_pgd_supports_longer_iteration_counts():
    operator = GaussianBlurOperator(kernel_size=11, sigma=2.0)
    model = SharedUnrolledPGD(operator=operator, n_iter=2, features=4)
    image = torch.rand(1, 1, 16, 16)
    blurry = make_blurry_observation(operator, image, noise_var=0.001)

    short_output = model(blurry)
    long_output = model(blurry, n_iter=4)

    assert short_output.shape == image.shape
    assert long_output.shape == image.shape
    assert all(step > 0 for step in model.positive_steps())


def test_make_blurry_observation_deterministic_with_manual_seed():
    operator = GaussianBlurOperator(kernel_size=11, sigma=2.0)
    image = torch.rand(1, 1, 16, 16)

    torch.manual_seed(123)
    first = make_blurry_observation(operator, image, noise_var=0.001)
    torch.manual_seed(123)
    second = make_blurry_observation(operator, image, noise_var=0.001)

    assert torch.allclose(first, second)
    assert first.shape == image.shape
    assert torch.min(first) >= 0.0
    assert torch.max(first) <= 1.0
