"""
Hard Physiological Constraints for TBI Neural ODE Models

These constraints are enforced as projection layers (not soft penalties),
ensuring model outputs never violate known physiological identities or bounds.
"""

from .monro_kellie import MonroKellieConstraint
from .cerebral_perfusion import CerebralPerfusionConstraint
from .physiological_bounds import PhysiologicalBounds

__all__ = [
    "MonroKellieConstraint",
    "CerebralPerfusionConstraint",
    "PhysiologicalBounds",
]
