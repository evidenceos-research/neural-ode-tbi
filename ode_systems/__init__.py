"""
ODE Systems for TBI Physiological Modeling

Mechanistic right-hand-side definitions encoding known TBI physiology:
- ICP dynamics (Monro-Kellie derived)
- Cerebral autoregulation (pressure-flow coupling)
- Biomarker kinetics (GFAP/UCH-L1 release and clearance)
- Multi-compartment integration
"""

from .icp_dynamics import ICPDynamics
from .cerebral_autoregulation import CerebralAutoregulation
from .biomarker_kinetics import BiomarkerKinetics
from .multi_compartment import MultiCompartmentTBI

__all__ = [
    "ICPDynamics",
    "CerebralAutoregulation",
    "BiomarkerKinetics",
    "MultiCompartmentTBI",
]
