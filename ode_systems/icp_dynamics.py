"""
ICP Dynamics ODE System

Models intracranial pressure dynamics based on the Monro-Kellie doctrine:
  V_brain + V_blood + V_csf + V_lesion = V_cranium (constant)

State variables:
  - ICP (mmHg)
  - CSF volume proxy (mL)
  - Cerebral blood volume proxy (mL)

The ODE encodes:
  dICP/dt = (1/C) * (Q_production - Q_absorption - Q_outflow + Q_edema)
  where C = intracranial compliance (pressure-volume relationship)
"""

import torch
import torch.nn as nn


class ICPDynamics(nn.Module):
    """Mechanistic ICP dynamics with optional neural augmentation."""

    def __init__(self, augment_dim: int = 0):
        super().__init__()
        # Physiological parameters (learnable within bounds)
        self.log_compliance = nn.Parameter(torch.tensor(0.0))  # log(C) for positivity
        self.log_csf_production = nn.Parameter(torch.tensor(-1.6))  # ~0.2 mL/min baseline
        self.log_absorption_coeff = nn.Parameter(torch.tensor(-2.3))  # absorption rate
        self.log_outflow_resistance = nn.Parameter(torch.tensor(1.6))  # R_out

        # Optional neural augmentation for unmodeled dynamics
        self.augment_dim = augment_dim
        if augment_dim > 0:
            self.augment_net = nn.Sequential(
                nn.Linear(3 + augment_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 16),
                nn.Tanh(),
                nn.Linear(16, 3 + augment_dim),
            )
        else:
            self.augment_net = None

    @property
    def compliance(self) -> torch.Tensor:
        return torch.exp(self.log_compliance).clamp(min=0.01, max=5.0)

    @property
    def csf_production_rate(self) -> torch.Tensor:
        return torch.exp(self.log_csf_production).clamp(min=0.05, max=1.0)

    @property
    def absorption_coeff(self) -> torch.Tensor:
        return torch.exp(self.log_absorption_coeff).clamp(min=0.01, max=1.0)

    @property
    def outflow_resistance(self) -> torch.Tensor:
        return torch.exp(self.log_outflow_resistance).clamp(min=1.0, max=50.0)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Compute dstate/dt.

        Args:
            t: Current time (scalar tensor)
            state: [..., 3+augment_dim] tensor
                   state[..., 0] = ICP (mmHg)
                   state[..., 1] = CSF volume proxy (mL)
                   state[..., 2] = Cerebral blood volume proxy (mL)

        Returns:
            dstate/dt with same shape as state
        """
        icp = state[..., 0:1]
        v_csf = state[..., 1:2]
        v_cbv = state[..., 2:3]

        # CSF dynamics
        q_prod = self.csf_production_rate
        q_absorb = self.absorption_coeff * torch.relu(icp - 5.0)  # absorption above ~5 mmHg
        q_outflow = (icp - 0.0) / self.outflow_resistance  # outflow against venous pressure (~0)

        # Pressure-volume: dICP/dt = (1/C) * net_volume_change
        net_volume_rate = q_prod - q_absorb - q_outflow
        d_icp = net_volume_rate / self.compliance

        # CSF volume tracks production minus absorption
        d_v_csf = q_prod - q_absorb

        # CBV changes driven by autoregulation (simplified)
        d_v_cbv = -0.01 * (v_cbv - 50.0)  # relaxation toward baseline ~50 mL

        d_phys = torch.cat([d_icp, d_v_csf, d_v_cbv], dim=-1)

        # Neural augmentation
        if self.augment_net is not None:
            aug = self.augment_net(state)
            d_phys = d_phys + aug[..., :3]
            d_aug = aug[..., 3:]
            return torch.cat([d_phys, d_aug], dim=-1)

        return d_phys

    def get_param_summary(self) -> dict:
        """Return current parameter values for logging."""
        return {
            "compliance": self.compliance.item(),
            "csf_production_rate": self.csf_production_rate.item(),
            "absorption_coeff": self.absorption_coeff.item(),
            "outflow_resistance": self.outflow_resistance.item(),
        }
