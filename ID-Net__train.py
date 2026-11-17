import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from crip.io import imreadTiff, imwriteTiff


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, channel=64, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, channel)
        self.down1 = Down(channel, channel * 2)
        self.down2 = Down(channel * 2, channel * 4)
        factor = 2 if bilinear else 1
        self.down3 = Down(channel * 4, channel * 8 // factor)

        self.up1 = Up(channel * 8, channel * 4 // factor, bilinear)
        self.up2 = Up(channel * 4, channel * 2 // factor, bilinear)
        self.up3 = Up(channel * 2, channel, bilinear)
        self.outc = OutConv(channel, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


def create_ori_noise(shape, input_channels=64, device="cpu"):
    h, w = shape
    z = torch.randn(1, input_channels, h, w, device=device)
    z.requires_grad_(False)
    return z


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=0.5):
        super().__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (torch.mean(h_tv) + torch.mean(w_tv))


def run_3d_dip(
    y_noisy: torch.Tensor,
    x_n2n: torch.Tensor,
    max_iter: int = 30000,
    lr: float = 4e-5,
    tv_weight: float = 1e-4,
    device: str = "cuda",
    input_channels: int = 64,
    ema_decay: float = 0.8,
    ema_start_iter: int = 2000,
    target_iters=None,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    y_noisy = y_noisy.to(device)
    x_n2n = x_n2n.to(device)

    if target_iters is None:
        target_iters = list(range(4000, max_iter + 1, 500))

    shape = x_n2n.shape[-2:]
    z = create_ori_noise(shape, input_channels=input_channels, device=device)

    net = UNet(n_channels=input_channels, n_classes=1).to(device)
    tv_loss = TVLoss(tv_weight)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)

    outputs = []
    ema_output = None
    target_iter_set = set(target_iters)

    pbar = tqdm(range(max_iter), desc="optimization")
    for it in pbar:
        optimizer.zero_grad()

        x_hat = net(z)

        data_loss = F.mse_loss(x_hat, x_n2n)
        tv = tv_loss(x_hat)
        loss = data_loss + tv

        loss.backward()
        optimizer.step()
        scheduler.step()

        iter_num = it + 1
        with torch.no_grad():
            current = x_hat.detach()
            if iter_num >= ema_start_iter:
                if ema_output is None:
                    ema_output = current.clone()
                else:
                    ema_output.mul_(ema_decay).add_(current, alpha=1.0 - ema_decay)

            if iter_num in target_iter_set:
                output_to_save = ema_output if ema_output is not None else current
                outputs.append(output_to_save.detach().cpu().clone())
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.6f}",
                        "data": f"{data_loss.item():.6f}",
                        "tv": f"{tv.item():.6f}",
                        "iter": iter_num,
                    }
                )

    return outputs


if __name__ == "__main__":
    y_noisy = torch.from_numpy(imreadTiff("low_dose_70ua_recon.tif")).to(torch.float32)
    x_n2n = torch.from_numpy(imreadTiff("low_dose_70ua_recon_my.tif")).to(torch.float32)

    target_iters = list(range(2000, 10001, 500))
    recon_by_iter = {iter_num: torch.zeros_like(x_n2n) for iter_num in target_iters}

    for slice_idx in range(x_n2n.shape[0]):
        output_slice = run_3d_dip(
            y_noisy[slice_idx].reshape((1, 1, *x_n2n.shape[-2:])),
            x_n2n[slice_idx].reshape((1, 1, *x_n2n.shape[-2:])),
            max_iter=10000,
            tv_weight=1e-7,
            target_iters=target_iters,
        )

        for iter_num, recon_slice in zip(target_iters, output_slice):
            recon_by_iter[iter_num][slice_idx] = recon_slice.squeeze(0).squeeze(0)

    for iter_num, recon in recon_by_iter.items():
        imwriteTiff(
            recon.data.cpu().numpy(),
            f"low_dose_70ua_recon_EMA_{iter_num}iters.tif",
        )
