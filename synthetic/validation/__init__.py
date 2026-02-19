"""
Synthetic Data Validation Metrics

Three-gate validation framework:
  Gate 1: Statistical Fidelity (KS tests, correlation preservation)
  Gate 2: Downstream Utility (TSTR/TRTS paradigm)
  Gate 3: Constraint Compliance (physics violation rates)
"""

from .fidelity_metrics import compute_fidelity_report
from .constraint_compliance import compute_constraint_compliance

__all__ = [
    "compute_fidelity_report",
    "compute_constraint_compliance",
]
