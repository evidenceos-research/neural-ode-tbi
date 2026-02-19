"""
Hybrid Neural ODE for TBI

Combines mechanistic ODE systems with neural network augmentation.
The mechanistic component encodes known physiology; the neural component
learns residual dynamics from data.

Architecture:
  1. Encoder: maps irregular observations to initial latent state
  2. ODE: MultiCompartmentTBI + neural augmentation
  3. Decoder: maps latent state to observation space
  4. Constraint projection: enforces hard physiological bounds after each step
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint

from ..ode_systems.multi_compartment import MultiCompartmentTBI
from ..constraints.physiological_bounds import PhysiologicalBounds
from ..constraints.monro_kellie import MonroKellieConstraint


class Encoder(nn.Module):
    """Maps observed clinical variables to ODE initial state."""

    def __init__(self, obs_dim: int, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Decoder(nn.Module):
    """Maps ODE state back to observation space."""

    def __init__(self, state_dim: int, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, obs_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class HybridNeuralODE(nn.Module):
    """
    Physics-informed hybrid Neural ODE for TBI trajectory prediction.

    Training loop:
      1. Encode first observation -> initial state
      2. Integrate ODE over time points
      3. Project each state through constraints
      4. Decode to observation space
      5. Compute loss against observed values
    """

    def __init__(
        self,
        obs_dim: int = 7,
        augment_dim: int = 4,
        solver: str = "dopri5",
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ):
        super().__init__()
        self.state_dim = MultiCompartmentTBI.STATE_DIM + augment_dim
        self.augment_dim = augment_dim

        self.encoder = Encoder(obs_dim, self.state_dim)
        self.ode_func = MultiCompartmentTBI(augment_dim=augment_dim)
        self.decoder = Decoder(self.state_dim, obs_dim)

        # Hard constraint projections
        self.bounds = PhysiologicalBounds()
        self.monro_kellie = MonroKellieConstraint()

        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def forward(
        self,
        first_obs: torch.Tensor,
        time_points: torch.Tensor,
        map_mmhg: torch.Tensor | None = None,
    ) -> dict:
        """
        Forward pass: encode -> integrate -> project -> decode.

        Args:
            first_obs: [batch, obs_dim] initial observation
            time_points: [T] time points to evaluate (hours since injury)
            map_mmhg: [batch] or scalar MAP input

        Returns:
            Dict with predicted observations, states, and constraint info.
        """
        # Encode initial state
        z0 = self.encoder(first_obs)

        # Wrap ODE func to pass MAP
        def ode_wrapper(t, z):
            return self.ode_func(t, z, map_mmhg=map_mmhg)

        # Integrate
        trajectories = odeint(
            ode_wrapper,
            z0,
            time_points,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )  # [T, batch, state_dim]

        # Apply hard constraint projections at each time step
        projected = []
        violations = []
        for t_idx in range(trajectories.shape[0]):
            state_t = trajectories[t_idx]
            state_t = self.bounds(state_t)
            state_t = self.monro_kellie(state_t)
            projected.append(state_t)
            violations.append(self.bounds.violation(state_t).item())

        projected_trajectories = torch.stack(projected, dim=0)

        # Decode to observation space
        predictions = self.decoder(projected_trajectories)

        return {
            "predictions": predictions,  # [T, batch, obs_dim]
            "states": projected_trajectories,  # [T, batch, state_dim]
            "raw_states": trajectories,  # [T, batch, state_dim] (pre-projection)
            "constraint_violations": violations,  # [T] list of violation magnitudes
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        constraint_weight: float = 0.1,
        raw_states: torch.Tensor | None = None,
    ) -> dict:
        """
        Compute training loss with optional constraint penalty.

        Args:
            predictions: [T, batch, obs_dim]
            targets: [T, batch, obs_dim]
            mask: [T, batch, obs_dim] binary mask for observed values
            constraint_weight: weight for constraint violation penalty
            raw_states: pre-projection states for computing soft penalty

        Returns:
            Dict with total_loss, reconstruction_loss, constraint_loss.
        """
        if mask is not None:
            recon_loss = ((predictions - targets) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        else:
            recon_loss = ((predictions - targets) ** 2).mean()

        constraint_loss = torch.tensor(0.0, device=predictions.device)
        if raw_states is not None:
            constraint_loss = self.bounds.violation(raw_states)

        total_loss = recon_loss + constraint_weight * constraint_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "constraint_loss": constraint_loss,
        }
