"""
Biomarker Kinetics ODE System

Models the release, distribution, and clearance of TBI blood biomarkers:
  - GFAP (glial fibrillary acidic protein) — astrocytic injury marker
  - UCH-L1 (ubiquitin C-terminal hydrolase L1) — neuronal injury marker

Kinetic model:
  dC/dt = k_release * injury_signal(t) - k_clearance * C

Where:
  C = plasma concentration (pg/mL)
  k_release = release rate from damaged tissue
  k_clearance = first-order elimination rate
  injury_signal(t) = time-varying injury severity proxy

Canonical thresholds (Abbott i-STAT / NINDS):
  GFAP >= 30 pg/mL  -> CT indicated
  UCH-L1 >= 360 pg/mL -> CT indicated
"""

import torch
import torch.nn as nn


# Canonical thresholds — single source of truth for Python side
GFAP_CT_THRESHOLD = 30.0  # pg/mL
UCHL1_CT_THRESHOLD = 360.0  # pg/mL


class BiomarkerKinetics(nn.Module):
    """
    Two-analyte biomarker kinetics ODE.

    State: [GFAP_concentration, UCH-L1_concentration]
    """

    def __init__(self, augment_dim: int = 0):
        super().__init__()
        # GFAP kinetics
        self.log_k_release_gfap = nn.Parameter(torch.tensor(1.0))
        self.log_k_clearance_gfap = nn.Parameter(torch.tensor(-3.77))  # ~0.023/hr, t½ ~30h

        # UCH-L1 kinetics (faster clearance than GFAP)
        self.log_k_release_uchl1 = nn.Parameter(torch.tensor(1.5))
        self.log_k_clearance_uchl1 = nn.Parameter(torch.tensor(-2.45))  # ~0.086/hr, t½ ~8h

        # Injury signal decay (exponential decay of acute release)
        self.log_injury_decay = nn.Parameter(torch.tensor(-1.0))

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
    def k_release_gfap(self) -> torch.Tensor:
        return torch.exp(self.log_k_release_gfap).clamp(min=0.01, max=100.0)

    @property
    def k_clearance_gfap(self) -> torch.Tensor:
        return torch.exp(self.log_k_clearance_gfap).clamp(min=0.01, max=2.0)

    @property
    def k_release_uchl1(self) -> torch.Tensor:
        return torch.exp(self.log_k_release_uchl1).clamp(min=0.01, max=200.0)

    @property
    def k_clearance_uchl1(self) -> torch.Tensor:
        return torch.exp(self.log_k_clearance_uchl1).clamp(min=0.01, max=2.0)

    @property
    def half_life_gfap_hours(self) -> torch.Tensor:
        return torch.log(torch.tensor(2.0, device=self.k_clearance_gfap.device)) / self.k_clearance_gfap

    @property
    def half_life_uchl1_hours(self) -> torch.Tensor:
        return torch.log(torch.tensor(2.0, device=self.k_clearance_uchl1.device)) / self.k_clearance_uchl1

    @property
    def injury_decay(self) -> torch.Tensor:
        return torch.exp(self.log_injury_decay).clamp(min=0.01, max=5.0)

    def injury_signal(self, t: torch.Tensor) -> torch.Tensor:
        """Time-decaying injury release signal."""
        return torch.exp(-self.injury_decay * t)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time since injury (hours)
            state: [..., 2+augment_dim]
                   state[..., 0] = GFAP concentration (pg/mL)
                   state[..., 1] = UCH-L1 concentration (pg/mL)

        Returns:
            dstate/dt
        """
        gfap = state[..., 0:1]
        uchl1 = state[..., 1:2]

        signal = self.injury_signal(t)

        d_gfap = self.k_release_gfap * signal - self.k_clearance_gfap * gfap
        d_uchl1 = self.k_release_uchl1 * signal - self.k_clearance_uchl1 * uchl1

        d_state = torch.cat([d_gfap, d_uchl1], dim=-1)

        if self.augment_net is not None:
            d_state = d_state + self.augment_net(state)

        return d_state

    def evaluate_thresholds(self, gfap: float, uchl1: float) -> dict:
        """Evaluate biomarker concentrations against canonical thresholds."""
        return {
            "gfap_pg_ml": gfap,
            "uchl1_pg_ml": uchl1,
            "gfap_positive": gfap >= GFAP_CT_THRESHOLD,
            "uchl1_positive": uchl1 >= UCHL1_CT_THRESHOLD,
            "ct_indicated": gfap >= GFAP_CT_THRESHOLD or uchl1 >= UCHL1_CT_THRESHOLD,
            "biomarker_negative": gfap < GFAP_CT_THRESHOLD and uchl1 < UCHL1_CT_THRESHOLD,
        }

    def get_param_summary(self) -> dict:
        return {
            "k_release_gfap": self.k_release_gfap.item(),
            "k_clearance_gfap": self.k_clearance_gfap.item(),
            "half_life_gfap_hours": self.half_life_gfap_hours.item(),
            "k_release_uchl1": self.k_release_uchl1.item(),
            "k_clearance_uchl1": self.k_clearance_uchl1.item(),
            "half_life_uchl1_hours": self.half_life_uchl1_hours.item(),
            "injury_decay": self.injury_decay.item(),
        }
