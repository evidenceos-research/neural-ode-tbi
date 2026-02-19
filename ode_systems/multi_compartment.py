"""
Multi-Compartment TBI ODE System

Integrates ICP dynamics, cerebral autoregulation, and biomarker kinetics
into a single coupled ODE system for joint trajectory prediction.

State vector layout:
  [ICP, V_csf, V_cbv, CVR, CBF, GFAP, UCH-L1, ...augment]

This is the primary ODE system used by the hybrid Neural ODE model.
"""

import torch
import torch.nn as nn

from .icp_dynamics import ICPDynamics
from .cerebral_autoregulation import CerebralAutoregulation
from .biomarker_kinetics import BiomarkerKinetics


class MultiCompartmentTBI(nn.Module):
    """
    Coupled multi-compartment ODE for TBI patient trajectory modeling.

    Combines:
      - ICP dynamics (3 states)
      - Cerebral autoregulation (2 states)
      - Biomarker kinetics (2 states)
      Total: 7 mechanistic states + optional augmentation dims
    """

    STATE_DIM = 7
    STATE_NAMES = ["ICP", "V_csf", "V_cbv", "CVR", "CBF", "GFAP", "UCH_L1"]

    def __init__(self, augment_dim: int = 0):
        super().__init__()
        self.augment_dim = augment_dim

        self.icp = ICPDynamics(augment_dim=0)
        self.autoreg = CerebralAutoregulation(augment_dim=0)
        self.biomarker = BiomarkerKinetics(augment_dim=0)

        # Cross-compartment coupling (learnable)
        self.icp_to_biomarker_gain = nn.Parameter(torch.tensor(0.1))

        # Global neural augmentation across all compartments
        if augment_dim > 0:
            self.global_augment = nn.Sequential(
                nn.Linear(self.STATE_DIM + augment_dim, 32),
                nn.Tanh(),
                nn.Linear(32, self.STATE_DIM + augment_dim),
            )
        else:
            self.global_augment = None

    def forward(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        map_mmhg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute dstate/dt for the full coupled system.

        Args:
            t: Current time (hours since injury)
            state: [..., 7+augment_dim]
            map_mmhg: External MAP input. If None, uses default 80 mmHg.

        Returns:
            dstate/dt
        """
        if map_mmhg is None:
            map_mmhg = torch.tensor(80.0, device=state.device, dtype=state.dtype)

        icp_state = state[..., 0:3]   # ICP, V_csf, V_cbv
        ar_state = state[..., 3:5]    # CVR, CBF
        bm_state = state[..., 5:7]    # GFAP, UCH-L1

        current_icp = state[..., 0:1]

        # Compute individual compartment derivatives
        d_icp = self.icp(t, icp_state)
        d_ar = self.autoreg(t, ar_state, map_mmhg=map_mmhg, icp_mmhg=current_icp)
        d_bm = self.biomarker(t, bm_state)

        # Cross-compartment coupling: elevated ICP increases biomarker release
        icp_excess = torch.relu(current_icp - 22.0)  # above treatment threshold
        coupling_boost = self.icp_to_biomarker_gain * icp_excess
        d_bm = d_bm + coupling_boost.expand_as(d_bm)

        d_state = torch.cat([d_icp, d_ar, d_bm], dim=-1)

        # Global augmentation
        if self.global_augment is not None:
            d_state_full = torch.cat(
                [d_state, torch.zeros_like(state[..., self.STATE_DIM:])], dim=-1
            )
            d_state_full = d_state_full + self.global_augment(state)
            return d_state_full

        return d_state

    def get_param_summary(self) -> dict:
        summary = {"icp_to_biomarker_gain": self.icp_to_biomarker_gain.item()}
        summary.update({f"icp.{k}": v for k, v in self.icp.get_param_summary().items()})
        summary.update({f"autoreg.{k}": v for k, v in self.autoreg.get_param_summary().items()})
        summary.update({f"biomarker.{k}": v for k, v in self.biomarker.get_param_summary().items()})
        return summary

    @staticmethod
    def default_initial_state(
        batch_size: int = 1,
        augment_dim: int = 0,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Return clinically plausible initial state."""
        state = torch.zeros(batch_size, MultiCompartmentTBI.STATE_DIM + augment_dim, device=device)
        state[:, 0] = 12.0   # ICP ~12 mmHg (normal)
        state[:, 1] = 120.0  # V_csf ~120 mL
        state[:, 2] = 50.0   # V_cbv ~50 mL
        state[:, 3] = 1.6    # CVR ~1.6 (arbitrary units)
        state[:, 4] = 50.0   # CBF ~50 mL/100g/min
        state[:, 5] = 5.0    # GFAP ~5 pg/mL (baseline)
        state[:, 6] = 50.0   # UCH-L1 ~50 pg/mL (baseline)
        return state
