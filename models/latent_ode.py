"""
Latent ODE Baseline

A standard latent ODE model (Rubanova et al., 2019) without mechanistic priors.
Used as a comparator to demonstrate the value of physics-informed constraints.

Architecture:
  1. GRU-based encoder for irregular time series
  2. Fully neural ODE in latent space
  3. Linear decoder to observation space
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint


class LatentODEFunc(nn.Module):
    """Fully neural ODE function (no mechanistic priors)."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GRUEncoder(nn.Module):
    """GRU encoder for irregularly sampled time series."""

    def __init__(self, obs_dim: int, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        self.gru = nn.GRU(obs_dim + 1, hidden_dim, batch_first=True)  # +1 for time delta
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        observations: torch.Tensor,
        time_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            observations: [batch, T, obs_dim]
            time_points: [batch, T] or [T]

        Returns:
            z0: [batch, latent_dim]
        """
        if time_points.dim() == 1:
            time_points = time_points.unsqueeze(0).expand(observations.shape[0], -1)

        # Compute time deltas
        dt = torch.zeros_like(time_points)
        dt[:, 1:] = time_points[:, 1:] - time_points[:, :-1]

        # Concatenate observations with time deltas
        x = torch.cat([observations, dt.unsqueeze(-1)], dim=-1)

        # Process in reverse time order (encode from last to first)
        x_reversed = torch.flip(x, dims=[1])
        _, h = self.gru(x_reversed)
        z0 = self.fc(h.squeeze(0))
        return z0


class LatentODE(nn.Module):
    """
    Standard Latent ODE baseline (no physics priors).

    Used as comparator for the HybridNeuralODE to quantify
    the benefit of mechanistic constraints.
    """

    def __init__(
        self,
        obs_dim: int = 7,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        solver: str = "dopri5",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        self.encoder = GRUEncoder(obs_dim, hidden_dim, latent_dim)
        self.ode_func = LatentODEFunc(latent_dim)
        self.decoder = nn.Linear(latent_dim, obs_dim)

        self.solver = solver

    def forward(
        self,
        observations: torch.Tensor,
        obs_times: torch.Tensor,
        pred_times: torch.Tensor,
    ) -> dict:
        """
        Args:
            observations: [batch, T_obs, obs_dim]
            obs_times: [T_obs] observation time points
            pred_times: [T_pred] prediction time points

        Returns:
            Dict with predictions and latent states.
        """
        z0 = self.encoder(observations, obs_times)

        z_traj = odeint(
            self.ode_func,
            z0,
            pred_times,
            method=self.solver,
        )  # [T_pred, batch, latent_dim]

        predictions = self.decoder(z_traj)  # [T_pred, batch, obs_dim]

        return {
            "predictions": predictions,
            "latent_states": z_traj,
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict:
        if mask is not None:
            recon_loss = ((predictions - targets) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        else:
            recon_loss = ((predictions - targets) ** 2).mean()

        return {
            "total_loss": recon_loss,
            "reconstruction_loss": recon_loss,
        }
