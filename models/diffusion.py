from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from Nichol & Dhariwal (Improved DDPM).
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class Conv1dDenoiser(nn.Module):
    """
    GRU + Conv1D denoiser for time-series diffusion.

    Inputs:
        noisy_target: (B, L, D)
        context:      (B, context_size, D)
        t:            (B,)
    """

    def __init__(
        self,
        feature_dim: int,
        context_size: int,
        horizon_size: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.hidden_dim = hidden_dim

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.context_rnn = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
        )

        in_channels = feature_dim + hidden_dim * 2
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim * 2, feature_dim, kernel_size=3, padding=1),
        )

    def forward(self, noisy_target: torch.Tensor, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = noisy_target.shape

        t_emb = self.time_embed(t.float().view(-1, 1)).view(bsz, -1, 1).expand(-1, -1, length)

        _, h_n = self.context_rnn(context)
        ctx_emb = h_n[-1].view(bsz, -1, 1).expand(-1, -1, length)

        x = noisy_target.permute(0, 2, 1)
        x = torch.cat([x, ctx_emb, t_emb], dim=1)
        out = self.conv_net(x)
        return out.permute(0, 2, 1)


class TimeSeriesDiffusion(nn.Module):
    """
    DDPM-style diffusion model for time-series forecasting.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        diffusion_steps: int = 50,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.diffusion_steps = diffusion_steps
        self.beta_schedule = beta_schedule

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(diffusion_steps)
        else:
            betas = torch.linspace(beta_start, beta_end, diffusion_steps)

        alphas = 1.0 - betas
        self.register_buffer("alpha_hat", torch.cumprod(alphas, dim=0))
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)

    def forward(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Training loss: predict noise added to target.
        """
        bsz = target.shape[0]
        device = target.device

        t = torch.randint(0, self.diffusion_steps, (bsz,), device=device)
        noise = torch.randn_like(target)

        a_hat = self.alpha_hat[t].view(-1, 1, 1)
        noisy = torch.sqrt(a_hat) * target + torch.sqrt(1 - a_hat) * noise

        pred_noise = self.denoiser(noisy, context, t)
        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        horizon: int,
        feature_dim: int,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate samples conditioned on context.

        Returns:
            (B, num_samples, horizon, feature_dim)
        """
        self.eval()
        device = context.device
        bsz = context.shape[0]

        if num_samples > 1:
            context = context.repeat_interleave(num_samples, dim=0)

        curr = torch.randn((bsz * num_samples, horizon, feature_dim), device=device)

        for t_step in reversed(range(self.diffusion_steps)):
            t = torch.full((bsz * num_samples,), t_step, device=device, dtype=torch.long)
            pred_noise = self.denoiser(curr, context, t)

            a = self.alphas[t_step]
            a_h = self.alpha_hat[t_step]
            b = self.betas[t_step]

            z = torch.randn_like(curr) if t_step > 0 else torch.zeros_like(curr)
            curr = (1 / torch.sqrt(a)) * (
                curr - ((1 - a) / torch.sqrt(1 - a_h)) * pred_noise
            ) + torch.sqrt(b) * z

        curr = curr.view(bsz, num_samples, horizon, feature_dim)
        return curr
