"""
Physiological Bounds Constraint

Hard-clamp projection ensuring all state variables remain within
physiologically plausible ranges. This is the Python-side mirror of
the TypeScript `tbi-constraint-validator.ts`.

Applied as a post-integration projection step after each ODE solve.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class BoundSpec:
    """Specification for a single physiological bound."""
    name: str
    index: int
    min_val: float
    max_val: float
    unit: str


# Bounds aligned with tbi-constraint-validator.ts
DEFAULT_BOUNDS = [
    BoundSpec("ICP", 0, 0.0, 100.0, "mmHg"),
    BoundSpec("V_csf", 1, 0.0, 300.0, "mL"),
    BoundSpec("V_cbv", 2, 0.0, 150.0, "mL"),
    BoundSpec("CVR", 3, 0.1, 20.0, "a.u."),
    BoundSpec("CBF", 4, 0.0, 150.0, "mL/100g/min"),
    BoundSpec("GFAP", 5, 0.0, 50000.0, "pg/mL"),
    BoundSpec("UCH_L1", 6, 0.0, 50000.0, "pg/mL"),
]


class PhysiologicalBounds(nn.Module):
    """
    Hard-clamp projection for physiological state variables.

    Ensures all ODE outputs remain within plausible ranges after each
    integration step. This is a non-differentiable projection; for
    differentiable alternatives, use soft penalty terms during training.
    """

    def __init__(self, bounds: list[BoundSpec] | None = None):
        super().__init__()
        self.bounds = bounds or DEFAULT_BOUNDS

        mins = torch.zeros(len(self.bounds))
        maxs = torch.zeros(len(self.bounds))
        for b in self.bounds:
            mins[b.index] = b.min_val
            maxs[b.index] = b.max_val

        self.register_buffer("mins", mins)
        self.register_buffer("maxs", maxs)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Clamp state to physiological bounds.

        Args:
            state: [..., state_dim] tensor

        Returns:
            Clamped state tensor
        """
        n = min(state.shape[-1], len(self.bounds))
        projected = state.clone()
        projected[..., :n] = torch.clamp(
            state[..., :n],
            min=self.mins[:n],
            max=self.maxs[:n],
        )
        return projected

    def violation(self, state: torch.Tensor) -> torch.Tensor:
        """Compute total out-of-bounds violation magnitude."""
        n = min(state.shape[-1], len(self.bounds))
        below = torch.relu(self.mins[:n] - state[..., :n])
        above = torch.relu(state[..., :n] - self.maxs[:n])
        return (below + above).sum(dim=-1).mean()

    def get_violation_report(self, state: torch.Tensor) -> list[dict]:
        """Return per-variable violation details for a single state vector."""
        reports = []
        for b in self.bounds:
            if b.index >= state.shape[-1]:
                continue
            val = state[..., b.index].item() if state.dim() == 1 else state[..., b.index].mean().item()
            if val < b.min_val or val > b.max_val:
                reports.append({
                    "variable": b.name,
                    "value": val,
                    "min": b.min_val,
                    "max": b.max_val,
                    "unit": b.unit,
                    "violation": max(b.min_val - val, val - b.max_val),
                })
        return reports
