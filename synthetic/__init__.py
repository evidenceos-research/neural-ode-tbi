"""
Synthetic Data Generation for TBI Neural ODE Research

Physics-informed synthetic data generation using the mechanistic ODE systems
as generative models. Supports:
  - Physiological time series (ICP, MAP, CPP, CBF, CVR)
  - Biomarker kinetics (GFAP, UCH-L1, S100B, NfL)
  - Full cohort assembly with demographics and outcomes
  - Subgroup-conditioned generation (severity, age, context)
  - Validation metrics (fidelity, utility, constraint compliance)
"""

from .physiology_generator import PhysiologyGenerator
from .biomarker_generator import BiomarkerGenerator
from .cohort_generator import CohortGenerator

__all__ = [
    "PhysiologyGenerator",
    "BiomarkerGenerator",
    "CohortGenerator",
]
