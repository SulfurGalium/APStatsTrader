from __future__ import annotations

import joblib
import torch

from config import BETA_SCHEDULE
from models.diffusion import Conv1dDenoiser, TimeSeriesDiffusion


def save_model_and_scaler(model: TimeSeriesDiffusion, scaler, prefix: str) -> None:
    """
    Save model state and scaler for later reconstruction.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_dim": model.denoiser.feature_dim,
            "context_size": model.denoiser.context_size,
            "horizon_size": model.denoiser.horizon_size,
            "diffusion_steps": model.diffusion_steps,
            "beta_schedule": getattr(model, "beta_schedule", "linear"),
            "hidden_dim": model.denoiser.hidden_dim,
        },
        f"{prefix}.pth",
    )
    joblib.dump(scaler, f"{prefix}_scaler.pkl")
    print(f"Saved model to {prefix}.pth and scaler to {prefix}_scaler.pkl")


def load_model_and_scaler(
    model_prefix: str,
    device: torch.device,
    feature_dim: int = 6,
    context_size: int = 60,
    horizon_size: int = 12,
    use_cosine_beta: bool = False,
):
    """
    Load diffusion model and scaler from disk.
    """
    model_path = f"{model_prefix}.pth"
    scaler_path = f"{model_prefix}_scaler.pkl"

    ckpt = torch.load(model_path, map_location=device)

    diffusion_steps = ckpt.get("diffusion_steps", 50)
    saved_beta_schedule = ckpt.get("beta_schedule") or BETA_SCHEDULE
    beta_schedule = "cosine" if use_cosine_beta or saved_beta_schedule == "cosine" else "linear"
    hidden_dim = ckpt.get("hidden_dim", 256)
    feature_dim = ckpt.get("feature_dim", feature_dim)
    context_size = ckpt.get("context_size", context_size)
    horizon_size = ckpt.get("horizon_size", horizon_size)

    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    denoiser = Conv1dDenoiser(
        feature_dim=feature_dim,
        context_size=context_size,
        horizon_size=horizon_size,
        hidden_dim=hidden_dim,
    )
    model = TimeSeriesDiffusion(
        denoiser=denoiser,
        diffusion_steps=diffusion_steps,
        beta_schedule=beta_schedule,
    )
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    scaler = joblib.load(scaler_path)
    print(
        f"Loaded model {model_path}: feature_dim={feature_dim}, context_size={context_size}, "
        f"horizon_size={horizon_size}, diffusion_steps={diffusion_steps}, beta_schedule={beta_schedule}"
    )
    return model, scaler
