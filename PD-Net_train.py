from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from crip.io import imreadTiff, imwriteTiff



class Net(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: int = 48) -> None:
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)


def build_network(
    in_channels: int,
    hidden_channels: int,
) -> nn.Module:
    return Net(in_channels, hidden_channels)


def pair_downsampler(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if img.ndim != 4:
        raise ValueError(f"Expected BCHW input, got shape {tuple(img.shape)}")

    channels = img.shape[1]
    kernel1 = img.new_tensor([[[[0.0, 0.5], [0.5, 0.0]]]])
    kernel2 = img.new_tensor([[[[0.5, 0.0], [0.0, 0.5]]]])
    kernel1 = kernel1.repeat(channels, 1, 1, 1)
    kernel2 = kernel2.repeat(channels, 1, 1, 1)

    sub1 = F.conv2d(img, kernel1, stride=2, groups=channels)
    sub2 = F.conv2d(img, kernel2, stride=2, groups=channels)
    return sub1, sub2


def zsn2n_loss(
    model: nn.Module,
    noisy: torch.Tensor,
    lambda_residual: float = 1.0,
    lambda_consistency: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noisy1, noisy2 = pair_downsampler(noisy)

    denoised1 = noisy1 - model(noisy1)
    denoised2 = noisy2 - model(noisy2)

    loss_residual = 0.5 * (
        F.mse_loss(denoised1, noisy2) + F.mse_loss(denoised2, noisy1)
    )

    noisy_full_denoised = noisy - model(noisy)
    full_denoised1, full_denoised2 = pair_downsampler(noisy_full_denoised)

    loss_consistency = 0.5 * (
        F.mse_loss(denoised1, full_denoised1)
        + F.mse_loss(denoised2, full_denoised2)
    )

    loss = (
        lambda_residual * loss_residual
        + lambda_consistency * loss_consistency
    )
    return loss, loss_residual, loss_consistency


def normalize_to_unit_interval(
    image: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value_min = image.amin(dim=(-2, -1), keepdim=True)
    value_max = image.amax(dim=(-2, -1), keepdim=True)
    value_range = (value_max - value_min).clamp_min(1e-8)
    normalized = (image - value_min) / value_range
    return normalized, value_min, value_range


def pad_to_even(image: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    height, width = image.shape[-2:]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    return image, pad_h, pad_w


def remove_even_padding(
    image: torch.Tensor,
    pad_h: int,
    pad_w: int,
) -> torch.Tensor:
    if pad_h:
        image = image[..., :-pad_h, :]
    if pad_w:
        image = image[..., :, :-pad_w]
    return image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denoise_single_projection(
    noisy_np: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    projection_index: int,
) -> Tuple[np.ndarray, float, float, float]:
    if noisy_np.ndim != 2:
        raise ValueError(
            f"Each projection must be 2-D, got shape {noisy_np.shape}"
        )

    set_seed(args.seed + projection_index)

    noisy = torch.from_numpy(noisy_np.astype(np.float32, copy=False))
    noisy = noisy.unsqueeze(0).unsqueeze(0).to(device)
    noisy, pad_h, pad_w = pad_to_even(noisy)

    noisy_normalized, value_min, value_range = normalize_to_unit_interval(noisy)

    model = build_network(
        in_channels=1,
        hidden_channels=args.hidden_channels,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step,
        gamma=args.lr_gamma,
    )

    model.train()
    final_total = final_residual = final_consistency = float("nan")

    for step in range(1, args.num_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total, residual, consistency = zsn2n_loss(
            model,
            noisy_normalized,
            lambda_residual=args.lambda_residual,
            lambda_consistency=args.lambda_consistency,
        )
        total.backward()
        optimizer.step()
        scheduler.step()

        final_total = float(total.detach().item())
        final_residual = float(residual.detach().item())
        final_consistency = float(consistency.detach().item())

        if step == 1 or step % args.log_interval == 0 or step == args.num_steps:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Projection {projection_index:04d} | "
                f"Step {step:04d}/{args.num_steps} | "
                f"Loss={final_total:.8e} | "
                f"Lres={final_residual:.8e} | "
                f"Lcons={final_consistency:.8e} | "
                f"LR={current_lr:.3e}"
            )

    model.eval()
    with torch.no_grad():
        denoised_normalized = noisy_normalized - model(noisy_normalized)
        if not args.disable_clamp:
            denoised_normalized = denoised_normalized.clamp(0.0, 1.0)

        denoised = denoised_normalized * value_range + value_min
        denoised = remove_even_padding(denoised, pad_h, pad_w)
        denoised_np = denoised.squeeze(0).squeeze(0).cpu().numpy()

    if args.save_models:
        model_dir = Path(args.output_dir) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            model.state_dict(),
            model_dir / f"projection_{projection_index:04d}.pth",
        )

    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return denoised_np.astype(np.float32), final_total, final_residual, final_consistency


def load_projection_stack(path: str) -> np.ndarray:
    images = np.asarray(imreadTiff(path), dtype=np.float32)
    if images.ndim == 2:
        images = images[None, ...]
    if images.ndim != 3:
        raise ValueError(
            "Expected a 2-D TIFF or a 3-D TIFF stack shaped [N, H, W], "
            f"but got {images.shape}."
        )
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_tiff",
        type=str,
        default="proj_70ua.tif",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
    )
    parser.add_argument("--output_name", type=str, default="ProjPred.tif")
    parser.add_argument("--gpu_devices", type=str, default="1")
    parser.add_argument("--hidden_channels", type=int, default=48)
    parser.add_argument("--unet_features", type=int, default=96)

    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_step", type=int, default=1500)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--lambda_residual", type=float, default=1.0)
    parser.add_argument("--lambda_consistency", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable_clamp", action="store_true")
    parser.add_argument("--save_models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    projections = load_projection_stack(args.input_tiff)
    print(f"Input stack: {projections.shape}")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")

    denoised_stack = np.empty_like(projections, dtype=np.float32)
    log_rows = []
    start_time = time.time()

    # Strict zero-shot setting: initialize and optimize a fresh model for each image.
    for index, noisy_projection in enumerate(projections):
        image_start = time.time()
        denoised, total, residual, consistency = denoise_single_projection(
            noisy_np=noisy_projection,
            args=args,
            device=device,
            projection_index=index,
        )
        denoised_stack[index] = denoised

        elapsed = time.time() - image_start
        log_rows.append(
            {
                "projection": index,
                "final_loss": total,
                "final_residual_loss": residual,
                "final_consistency_loss": consistency,
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"Finished projection {index + 1}/{len(projections)} "
            f"in {elapsed:.2f} s"
        )

    output_path = output_dir / args.output_name
    imwriteTiff(denoised_stack, str(output_path))

    print(f"Denoised projections saved to: {output_path}")
    print(f"Total elapsed time: {time.time() - start_time:.2f} s")


if __name__ == "__main__":
    main()
