"""
Evaluation modules for TBI Neural ODE models.

- trajectory_metrics: MSE, MAE, R², per-variable tracking
- event_prediction: ICP crisis prediction, biomarker threshold crossing
- constraint_compliance: physiological bound violation rates
"""

from .trajectory_metrics import compute_trajectory_metrics
from .event_prediction import evaluate_event_prediction
from .constraint_compliance import evaluate_constraint_compliance

__all__ = [
    "compute_trajectory_metrics",
    "evaluate_event_prediction",
    "evaluate_constraint_compliance",
]
