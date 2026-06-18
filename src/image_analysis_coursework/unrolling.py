"""Part I.B(i): unrolled proximal gradient reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from image_analysis_coursework.a1 import download_dataset, load_ground_truth


class MicroscopyPatchDataset(Dataset[torch.Tensor]):
    """Load microscopy TIFFs as single-channel tensors with optional augmentation."""

    def __init__(
        self,
        root_dir: str | Path,
        augment: bool = True,
        patch_size: int | None = 128,
        max_images: int | None = None,
    ) -> None:
        self.image_paths = sorted(Path(root_dir).glob("*.tif"))
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
        if not self.image_paths:
            raise ValueError(f"No .tif images found in {root_dir}")
        self.augment = augment
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = load_ground_truth(self.image_paths[index])
        tensor = torch.from_numpy(image).float().unsqueeze(0)
        _, height, width = tensor.shape

        if self.patch_size is not None and height >= self.patch_size and width >= self.patch_size:
            if self.augment:
                top = random.randint(0, height - self.patch_size)
                left = random.randint(0, width - self.patch_size)
            else:
                top = (height - self.patch_size) // 2
                left = (width - self.patch_size) // 2
            tensor = tensor[:, top : top + self.patch_size, left : left + self.patch_size]

        if self.augment:
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=(2,))
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=(1,))
        return tensor


class GaussianBlurOperator:
    """Depthwise Gaussian blur operator with a symmetric adjoint."""

    def __init__(self, kernel_size: int, sigma: float, channels: int = 1, device: str | torch.device = "cpu") -> None:
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        self.device = torch.device(device)
        self.kernel = self._create_gaussian_kernel().to(self.device)

    def _create_gaussian_kernel(self) -> torch.Tensor:
        axis = torch.arange(self.kernel_size, dtype=torch.float32) - self.kernel_size // 2
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(self.channels, 1, 1, 1)

    def A(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel.to(x.device), padding=self.kernel_size // 2, groups=self.channels)

    def AT(self, y: torch.Tensor) -> torch.Tensor:
        # The Gaussian kernel is symmetric, so the adjoint is the same padded convolution.
        return F.conv2d(y, self.kernel.to(y.device), padding=self.kernel_size // 2, groups=self.channels)


class SimpleRegularizer(nn.Module):
    """Small CNN used as the learned proximal map in unrolled PGD."""

    def __init__(self, channels: int = 1, features: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=3, padding=1),
            nn.PReLU(features),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.PReLU(features),
            nn.Conv2d(features, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class UnrolledPGD(nn.Module):
    """K-step unrolled PGD network with iteration-specific proximal CNNs."""

    def __init__(self, operator: GaussianBlurOperator, n_iter: int = 6, channels: int = 1, features: int = 32) -> None:
        super().__init__()
        self.operator = operator
        self.n_iter = n_iter
        self.prox_nets = nn.ModuleList(SimpleRegularizer(channels, features) for _ in range(n_iter))
        self.raw_steps = nn.Parameter(torch.full((n_iter,), -2.0))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        x = self.operator.AT(y)
        for k in range(self.n_iter):
            step = F.softplus(self.raw_steps[k])
            gradient = self.operator.AT(self.operator.A(x) - y)
            x = self.prox_nets[k](x - step * gradient)
            x = torch.clamp(x, 0.0, 1.0)
        return x

    def positive_steps(self) -> list[float]:
        return [float(value) for value in F.softplus(self.raw_steps.detach()).cpu()]


class SharedUnrolledPGD(nn.Module):
    """Unrolled PGD network with one shared proximal CNN and one shared step size."""

    def __init__(self, operator: GaussianBlurOperator, n_iter: int = 6, channels: int = 1, features: int = 32) -> None:
        super().__init__()
        self.operator = operator
        self.n_iter = n_iter
        self.prox_net = SimpleRegularizer(channels, features)
        self.raw_step = nn.Parameter(torch.tensor(-2.0))

    def forward(self, y: torch.Tensor, n_iter: int | None = None) -> torch.Tensor:
        iterations = self.n_iter if n_iter is None else n_iter
        if iterations < 1:
            raise ValueError("n_iter must be at least 1")
        x = self.operator.AT(y)
        step = F.softplus(self.raw_step)
        for _ in range(iterations):
            gradient = self.operator.AT(self.operator.A(x) - y)
            x = self.prox_net(x - step * gradient)
            x = torch.clamp(x, 0.0, 1.0)
        return x

    def positive_steps(self) -> list[float]:
        return [float(F.softplus(self.raw_step.detach()).cpu())]


@dataclass(frozen=True)
class TrainingHistory:
    epochs: list[dict[str, float]]


@dataclass(frozen=True)
class EvaluationResult:
    blur_psnr: float
    blur_ssim: float
    recon_psnr: float
    recon_ssim: float
    num_images: int


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_blurry_observation(
    operator: GaussianBlurOperator,
    ground_truth: torch.Tensor,
    noise_var: float = 0.001,
) -> torch.Tensor:
    blurred = operator.A(ground_truth)
    noise = math.sqrt(noise_var) * torch.randn_like(blurred)
    return torch.clamp(blurred + noise, 0.0, 1.0)


def train_unrolled_pgd(
    model: nn.Module,
    train_loader: DataLoader[torch.Tensor],
    operator: GaussianBlurOperator,
    device: torch.device,
    epochs: int = 20,
    noise_var: float = 0.001,
    learning_rate: float = 1e-4,
) -> TrainingHistory:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for ground_truth in train_loader:
            ground_truth = ground_truth.to(device)
            blurry = make_blurry_observation(operator, ground_truth, noise_var=noise_var)
            reconstruction = model(blurry)
            loss = F.l1_loss(reconstruction, ground_truth)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        mean_loss = total_loss / max(1, len(train_loader))
        history.append({"epoch": float(epoch + 1), "loss": mean_loss})
        print(f"Epoch {epoch + 1}/{epochs} | L1 loss {mean_loss:.6f}")

    return TrainingHistory(epochs=history)


def evaluate_unrolled_pgd(
    model: nn.Module,
    test_loader: DataLoader[torch.Tensor],
    operator: GaussianBlurOperator,
    device: torch.device,
    noise_var: float = 0.001,
    comparison_path: str | Path | None = None,
    reconstruction_iterations: int | None = None,
) -> EvaluationResult:
    model.eval()
    blur_psnr = blur_ssim = recon_psnr = recon_ssim = 0.0
    num_images = 0
    first_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    with torch.no_grad():
        for ground_truth in test_loader:
            ground_truth = ground_truth.to(device)
            blurry = make_blurry_observation(operator, ground_truth, noise_var=noise_var)
            if reconstruction_iterations is None:
                reconstruction = model(blurry)
            else:
                reconstruction = model(blurry, n_iter=reconstruction_iterations)

            for index in range(ground_truth.shape[0]):
                gt_np = ground_truth[index, 0].detach().cpu().numpy()
                blur_np = blurry[index, 0].detach().cpu().numpy()
                recon_np = reconstruction[index, 0].detach().cpu().numpy()
                blur_psnr += float(peak_signal_noise_ratio(gt_np, blur_np, data_range=1.0))
                blur_ssim += float(structural_similarity(gt_np, blur_np, data_range=1.0))
                recon_psnr += float(peak_signal_noise_ratio(gt_np, recon_np, data_range=1.0))
                recon_ssim += float(structural_similarity(gt_np, recon_np, data_range=1.0))
                num_images += 1

            if first_batch is None:
                first_batch = (ground_truth[:1].cpu(), blurry[:1].cpu(), reconstruction[:1].cpu())

    if num_images == 0:
        raise ValueError("test_loader produced no images")

    if comparison_path is not None and first_batch is not None:
        _save_unrolling_comparison(*first_batch, output_path=Path(comparison_path))

    return EvaluationResult(
        blur_psnr=blur_psnr / num_images,
        blur_ssim=blur_ssim / num_images,
        recon_psnr=recon_psnr / num_images,
        recon_ssim=recon_ssim / num_images,
        num_images=num_images,
    )


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _prepare_loaders(
    data_dir: str | Path,
    batch_size: int,
    patch_size: int,
    max_train_images: int | None,
    max_test_images: int | None,
) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
    dataset_root = download_dataset(data_dir)
    train_dataset = MicroscopyPatchDataset(dataset_root / "01", augment=True, patch_size=patch_size, max_images=max_train_images)
    test_dataset = MicroscopyPatchDataset(dataset_root / "02", augment=False, patch_size=None, max_images=max_test_images)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    return train_loader, test_loader


def _save_checkpoint(path: Path, model: nn.Module, n_iter: int, features: int, extra: dict[str, Any]) -> None:
    positive_steps = model.positive_steps() if hasattr(model, "positive_steps") else []
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_iter": n_iter,
            "features": features,
            "positive_steps": positive_steps,
            **extra,
        },
        path,
    )


def run_unrolling_experiment(
    data_dir: str | Path,
    output_dir: str | Path,
    train_kernel_size: int,
    train_sigma: float,
    test_kernel_size: int,
    test_sigma: float,
    model_kind: str = "independent",
    epochs: int = 20,
    batch_size: int = 4,
    patch_size: int = 128,
    n_iter: int = 6,
    features: int = 32,
    learning_rate: float = 1e-4,
    noise_var: float = 0.001,
    max_train_images: int | None = None,
    max_test_images: int | None = None,
    device_name: str = "auto",
    seed: int = 42,
) -> tuple[TrainingHistory, EvaluationResult, dict[str, Path]]:
    """Train and evaluate an unrolled PGD model under configurable blur settings."""

    _seed_everything(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = _prepare_loaders(data_dir, batch_size, patch_size, max_train_images, max_test_images)

    device = choose_device(device_name)
    train_operator = GaussianBlurOperator(kernel_size=train_kernel_size, sigma=train_sigma, channels=1, device=device)
    test_operator = GaussianBlurOperator(kernel_size=test_kernel_size, sigma=test_sigma, channels=1, device=device)
    if model_kind == "independent":
        model: nn.Module = UnrolledPGD(operator=train_operator, n_iter=n_iter, channels=1, features=features).to(device)
    elif model_kind == "shared":
        model = SharedUnrolledPGD(operator=train_operator, n_iter=n_iter, channels=1, features=features).to(device)
    else:
        raise ValueError("model_kind must be 'independent' or 'shared'")

    history = train_unrolled_pgd(
        model,
        train_loader,
        train_operator,
        device,
        epochs=epochs,
        noise_var=noise_var,
        learning_rate=learning_rate,
    )

    model.operator = test_operator
    comparison_path = output_path / "comparison.png"
    evaluation = evaluate_unrolled_pgd(
        model,
        test_loader,
        test_operator,
        device,
        noise_var=noise_var,
        comparison_path=comparison_path,
    )

    files = {
        "checkpoint": output_path / "unrolled_pgd.pt",
        "history": output_path / "history.json",
        "metrics": output_path / "metrics.json",
        "comparison": comparison_path,
    }
    files["history"].write_text(json.dumps({"epochs": history.epochs}, indent=2) + "\n", encoding="utf-8")
    metrics: dict[str, Any] = {
        "model_kind": model_kind,
        "train_blur_kernel_size": train_kernel_size,
        "train_blur_sigma": train_sigma,
        "test_blur_kernel_size": test_kernel_size,
        "test_blur_sigma": test_sigma,
        "noise_var": noise_var,
        "noise_std": math.sqrt(noise_var),
        "n_iter": n_iter,
        "features": features,
        "epochs": epochs,
        "batch_size": batch_size,
        "patch_size": patch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        "max_train_images": max_train_images,
        "max_test_images": max_test_images,
        "learned_steps": model.positive_steps() if hasattr(model, "positive_steps") else [],
        "blur_psnr": evaluation.blur_psnr,
        "blur_ssim": evaluation.blur_ssim,
        "recon_psnr": evaluation.recon_psnr,
        "recon_ssim": evaluation.recon_ssim,
        "num_test_images": evaluation.num_images,
    }
    files["metrics"].write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _save_checkpoint(
        files["checkpoint"],
        model,
        n_iter=n_iter,
        features=features,
        extra={
            "model_kind": model_kind,
            "train_blur_kernel_size": train_kernel_size,
            "train_blur_sigma": train_sigma,
            "test_blur_kernel_size": test_kernel_size,
            "test_blur_sigma": test_sigma,
        },
    )
    return history, evaluation, files


def run_b2_training(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs/part_i/b2",
    epochs: int = 20,
    batch_size: int = 4,
    patch_size: int = 128,
    n_iter: int = 6,
    features: int = 32,
    learning_rate: float = 1e-4,
    noise_var: float = 0.001,
    max_train_images: int | None = None,
    max_test_images: int | None = None,
    device_name: str = "auto",
    seed: int = 42,
) -> tuple[TrainingHistory, EvaluationResult, dict[str, Path]]:
    """Part I.B(ii): train on stronger blur and test on the B(i) blur."""

    return run_unrolling_experiment(
        data_dir=data_dir,
        output_dir=output_dir,
        train_kernel_size=21,
        train_sigma=4.0,
        test_kernel_size=11,
        test_sigma=2.0,
        model_kind="independent",
        epochs=epochs,
        batch_size=batch_size,
        patch_size=patch_size,
        n_iter=n_iter,
        features=features,
        learning_rate=learning_rate,
        noise_var=noise_var,
        max_train_images=max_train_images,
        max_test_images=max_test_images,
        device_name=device_name,
        seed=seed,
    )


@dataclass(frozen=True)
class ConvergenceResult:
    rows: list[dict[str, float]]


def run_b3_convergence(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs/part_i/b3",
    epochs: int = 20,
    batch_size: int = 4,
    patch_size: int = 128,
    n_iter: int = 6,
    features: int = 32,
    learning_rate: float = 1e-4,
    noise_var: float = 0.001,
    max_train_images: int | None = None,
    max_test_images: int | None = 1,
    device_name: str = "auto",
    seed: int = 42,
) -> tuple[TrainingHistory, ConvergenceResult, dict[str, Path]]:
    """Part I.B(iii): train shared unrolling and evaluate longer iteration counts."""

    _seed_everything(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = _prepare_loaders(data_dir, batch_size, patch_size, max_train_images, max_test_images)
    device = choose_device(device_name)
    operator = GaussianBlurOperator(kernel_size=11, sigma=2.0, channels=1, device=device)
    model = SharedUnrolledPGD(operator=operator, n_iter=n_iter, channels=1, features=features).to(device)
    history = train_unrolled_pgd(
        model,
        train_loader,
        operator,
        device,
        epochs=epochs,
        noise_var=noise_var,
        learning_rate=learning_rate,
    )

    ground_truth = next(iter(test_loader)).to(device)
    blurry = make_blurry_observation(operator, ground_truth, noise_var=noise_var)
    iteration_counts = [n_iter, 4 * n_iter, 8 * n_iter, 16 * n_iter]
    rows: list[dict[str, float]] = []
    reconstructions: list[tuple[int, np.ndarray]] = []
    with torch.no_grad():
        for iterations in iteration_counts:
            reconstruction = model(blurry, n_iter=iterations)
            gt_np = ground_truth[0, 0].detach().cpu().numpy()
            recon_np = reconstruction[0, 0].detach().cpu().numpy()
            rows.append(
                {
                    "iterations": float(iterations),
                    "psnr": float(peak_signal_noise_ratio(gt_np, recon_np, data_range=1.0)),
                    "ssim": float(structural_similarity(gt_np, recon_np, data_range=1.0)),
                }
            )
            reconstructions.append((iterations, recon_np))

    result = ConvergenceResult(rows=rows)
    files = {
        "checkpoint": output_path / "shared_unrolled_pgd.pt",
        "history": output_path / "history.json",
        "metrics": output_path / "metrics.json",
        "plot": output_path / "convergence_psnr.png",
        "comparison": output_path / "comparison.png",
    }
    files["history"].write_text(json.dumps({"epochs": history.epochs}, indent=2) + "\n", encoding="utf-8")
    files["metrics"].write_text(
        json.dumps(
            {
                "model_kind": "shared",
                "blur_kernel_size": 11,
                "blur_sigma": 2.0,
                "noise_var": noise_var,
                "n_iter": n_iter,
                "features": features,
                "epochs": epochs,
                "batch_size": batch_size,
                "patch_size": patch_size,
                "learning_rate": learning_rate,
                "device": str(device),
                "max_train_images": max_train_images,
                "max_test_images": max_test_images,
                "learned_steps": model.positive_steps(),
                "convergence": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _save_checkpoint(files["checkpoint"], model, n_iter=n_iter, features=features, extra={"model_kind": "shared"})
    _save_convergence_plot(result, files["plot"])
    _save_convergence_comparison(ground_truth.cpu(), blurry.cpu(), reconstructions, files["comparison"])
    return history, result, files


def run_b1_training(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs/part_i/b1",
    epochs: int = 20,
    batch_size: int = 4,
    patch_size: int = 128,
    n_iter: int = 6,
    features: int = 32,
    learning_rate: float = 1e-4,
    noise_var: float = 0.001,
    max_train_images: int | None = None,
    max_test_images: int | None = None,
    device_name: str = "auto",
    seed: int = 42,
) -> tuple[TrainingHistory, EvaluationResult, dict[str, Path]]:
    return run_unrolling_experiment(
        data_dir=data_dir,
        output_dir=output_dir,
        train_kernel_size=11,
        train_sigma=2.0,
        test_kernel_size=11,
        test_sigma=2.0,
        model_kind="independent",
        epochs=epochs,
        batch_size=batch_size,
        patch_size=patch_size,
        n_iter=n_iter,
        features=features,
        learning_rate=learning_rate,
        noise_var=noise_var,
        max_train_images=max_train_images,
        max_test_images=max_test_images,
        device_name=device_name,
        seed=seed,
    )


def _save_unrolling_comparison(
    ground_truth: torch.Tensor,
    blurry: torch.Tensor,
    reconstruction: torch.Tensor,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Ground truth", ground_truth[0, 0].numpy()),
        ("Blurred + noise", blurry[0, 0].numpy()),
        ("Unrolled PGD", reconstruction[0, 0].numpy()),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, (title, image) in zip(axes, panels, strict=True):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_convergence_plot(result: ConvergenceResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterations = [row["iterations"] for row in result.rows]
    psnr_values = [row["psnr"] for row in result.rows]
    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.plot(iterations, psnr_values, marker="o")
    axis.set_xlabel("PGD iterations T")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Shared unrolled PGD convergence")
    axis.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_convergence_comparison(
    ground_truth: torch.Tensor,
    blurry: torch.Tensor,
    reconstructions: list[tuple[int, np.ndarray]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Ground truth", ground_truth[0, 0].numpy()),
        ("Blurred + noise", blurry[0, 0].numpy()),
        *[(f"T={iterations}", image) for iterations, image in reconstructions],
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), constrained_layout=True)
    for axis, (title, image) in zip(np.ravel(axes), panels, strict=True):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
