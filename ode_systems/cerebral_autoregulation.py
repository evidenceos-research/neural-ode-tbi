"""
Cerebral Autoregulation ODE System

Models the pressure-flow autoregulation mechanism that maintains
cerebral blood flow (CBF) across a range of cerebral perfusion pressures (CPP).

Key relationships:
  CPP = MAP - ICP
  CBF = CPP / CVR  (cerebrovascular resistance)
  Autoregulation adjusts CVR to maintain CBF ~50 mL/100g/min

State variables:
  - CVR (cerebrovascular resistance, arbitrary units)
  - CBF (cerebral blood flow, mL/100g/min)

Inputs (from other compartments or observations):
  - MAP (mean arterial pressure, mmHg)
  - ICP (intracranial pressure, mmHg)
"""

import torch
import torch.nn as nn


class CerebralAutoregulation(nn.Module):
    """
    Autoregulation ODE: CVR adjusts to maintain CBF near target.

    When autoregulation is intact, CVR increases with rising CPP and
    decreases with falling CPP. When impaired (e.g., severe TBI),
    CBF becomes pressure-passive.
    """

    def __init__(self, augment_dim: int = 0):
        super().__init__()
        # Target CBF (mL/100g/min) — learnable
        self.log_cbf_target = nn.Parameter(torch.tensor(3.91))  # ~50 mL/100g/min

        # Autoregulation gain (how fast CVR adjusts)
        self.log_autoreg_gain = nn.Parameter(torch.tensor(-1.0))

        # Autoregulation integrity index [0, 1]: 1 = intact, 0 = fully impaired
        self.autoreg_integrity_logit = nn.Parameter(torch.tensor(2.0))  # ~0.88

        # CVR relaxation time constant
        self.log_tau_cvr = nn.Parameter(torch.tensor(1.0))  # ~2.7 min

        self.augment_dim = augment_dim
        if augment_dim > 0:
            self.augment_net = nn.Sequential(
                nn.Linear(2 + augment_dim, 16),
                nn.Tanh(),
                nn.Linear(16, 2 + augment_dim),
            )
        else:
            self.augment_net = None

    @property
    def cbf_target(self) -> torch.Tensor:
        return torch.exp(self.log_cbf_target).clamp(min=20.0, max=80.0)

    @property
    def autoreg_gain(self) -> torch.Tensor:
        return torch.exp(self.log_autoreg_gain).clamp(min=0.01, max=5.0)

    @property
    def autoreg_integrity(self) -> torch.Tensor:
        return torch.sigmoid(self.autoreg_integrity_logit)

    @property
    def tau_cvr(self) -> torch.Tensor:
        return torch.exp(self.log_tau_cvr).clamp(min=0.5, max=30.0)

    def forward(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        map_mmhg: torch.Tensor,
        icp_mmhg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dstate/dt for autoregulation dynamics.

        Args:
            t: Current time
            state: [..., 2+augment_dim]
                   state[..., 0] = CVR
                   state[..., 1] = CBF
            map_mmhg: Mean arterial pressure (external input)
            icp_mmhg: Intracranial pressure (from ICP compartment)

        Returns:
            dstate/dt
        """
        cvr = state[..., 0:1]
        cbf = state[..., 1:2]

        # CPP = MAP - ICP (hard physiological identity)
        cpp = (map_mmhg - icp_mmhg).clamp(min=0.0)

        # Current CBF from Ohm's law: CBF = CPP / CVR
        cbf_current = cpp / cvr.clamp(min=0.1)

        # Autoregulation drive: adjust CVR to bring CBF toward target
        cbf_error = cbf_current - self.cbf_target
        autoreg_drive = self.autoreg_gain * cbf_error * self.autoreg_integrity

        # CVR dynamics: dCVR/dt = (autoreg_drive) / tau
        d_cvr = autoreg_drive / self.tau_cvr

        # CBF follows from updated CVR (algebraic, but we track it as state for smoothness)
        d_cbf = (cbf_current - cbf) / self.tau_cvr

        d_state = torch.cat([d_cvr, d_cbf], dim=-1)

        if self.augment_net is not None:
            d_state = d_state + self.augment_net(state)

        return d_state

    def get_param_summary(self) -> dict:
        return {
            "cbf_target": self.cbf_target.item(),
            "autoreg_gain": self.autoreg_gain.item(),
            "autoreg_integrity": self.autoreg_integrity.item(),
            "tau_cvr": self.tau_cvr.item(),
        }
