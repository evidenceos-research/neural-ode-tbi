"""
Monro-Kellie Doctrine Constraint

The total intracranial volume is approximately constant:
  V_brain + V_blood + V_csf + V_lesion ≈ V_cranium

This constraint projects ODE outputs to satisfy the volume conservation identity.
"""

import torch
import torch.nn as nn


class MonroKellieConstraint(nn.Module):
    """
    Projects state to satisfy Monro-Kellie volume conservation.

    Enforces: V_csf + V_cbv <= V_cranium - V_brain_fixed
    If the sum exceeds the budget, proportionally scales both down.
    """

    def __init__(self, v_cranium: float = 1400.0, v_brain_fixed: float = 1200.0):
        super().__init__()
        self.register_buffer("volume_budget", torch.tensor(v_cranium - v_brain_fixed))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Project state to satisfy volume constraint.

        Args:
            state: [..., >=3] where state[..., 1] = V_csf, state[..., 2] = V_cbv

        Returns:
            Projected state with volume conservation enforced.
        """
        v_csf = state[..., 1:2]
        v_cbv = state[..., 2:3]

        total = v_csf + v_cbv
        excess = torch.relu(total - self.volume_budget)

        # Proportional scaling when over budget
        scale = torch.where(
            total > 0,
            (total - excess) / total.clamp(min=1e-6),
            torch.ones_like(total),
        )

        projected = state.clone()
        projected[..., 1:2] = v_csf * scale
        projected[..., 2:3] = v_cbv * scale

        return projected

    def violation(self, state: torch.Tensor) -> torch.Tensor:
        """Compute volume conservation violation magnitude."""
        total = state[..., 1] + state[..., 2]
        return torch.relu(total - self.volume_budget).mean()
