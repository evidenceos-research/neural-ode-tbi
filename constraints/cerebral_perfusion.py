"""
Cerebral Perfusion Pressure Constraint

Hard identity: CPP = MAP - ICP

This module enforces the CPP identity and flags critical thresholds:
  - CPP < 60 mmHg: critical (inadequate perfusion)
  - CPP > 70 mmHg: target range for most adult TBI patients
  - ICP > 22 mmHg: treatment threshold (BTF guidelines)
"""

import torch
import torch.nn as nn


# Clinical thresholds
ICP_TREATMENT_THRESHOLD = 22.0  # mmHg
CPP_CRITICAL_LOW = 60.0  # mmHg
CPP_TARGET_LOW = 60.0  # mmHg
CPP_TARGET_HIGH = 70.0  # mmHg


class CerebralPerfusionConstraint(nn.Module):
    """Enforces CPP = MAP - ICP identity and flags critical states."""

    def __init__(self):
        super().__init__()

    def compute_cpp(
        self, map_mmhg: torch.Tensor, icp_mmhg: torch.Tensor
    ) -> torch.Tensor:
        """Compute CPP from MAP and ICP (hard identity)."""
        return (map_mmhg - icp_mmhg).clamp(min=0.0)

    def forward(
        self,
        state: torch.Tensor,
        map_mmhg: torch.Tensor,
    ) -> dict:
        """
        Evaluate perfusion constraints.

        Args:
            state: [..., >=1] where state[..., 0] = ICP
            map_mmhg: Mean arterial pressure

        Returns:
            Dict with cpp, icp_crisis flags, cpp_critical flags.
        """
        icp = state[..., 0]
        cpp = self.compute_cpp(map_mmhg, icp)

        return {
            "cpp": cpp,
            "icp_crisis": (icp > ICP_TREATMENT_THRESHOLD).float(),
            "cpp_critical": (cpp < CPP_CRITICAL_LOW).float(),
            "cpp_in_target": ((cpp >= CPP_TARGET_LOW) & (cpp <= CPP_TARGET_HIGH)).float(),
        }

    def violation(self, state: torch.Tensor, map_mmhg: torch.Tensor) -> torch.Tensor:
        """Compute total constraint violation score."""
        result = self.forward(state, map_mmhg)
        return result["icp_crisis"].mean() + result["cpp_critical"].mean()
