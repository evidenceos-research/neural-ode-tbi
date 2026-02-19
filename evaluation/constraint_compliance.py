"""
Constraint compliance evaluation.

Measures how well model predictions respect hard physiological constraints.
This is a first-class evaluation metric — not just a training regularizer.
"""

import numpy as np


PHYSIOLOGICAL_BOUNDS = {
    "ICP": (0.0, 100.0, "mmHg"),
    "V_csf": (0.0, 300.0, "mL"),
    "V_cbv": (0.0, 150.0, "mL"),
    "CVR": (0.1, 20.0, "a.u."),
    "CBF": (0.0, 150.0, "mL/100g/min"),
    "GFAP": (0.0, 50000.0, "pg/mL"),
    "UCH_L1": (0.0, 50000.0, "pg/mL"),
}

VARIABLE_NAMES = list(PHYSIOLOGICAL_BOUNDS.keys())


def evaluate_constraint_compliance(
    states: np.ndarray,
    variable_names: list[str] | None = None,
) -> dict:
    """
    Evaluate physiological constraint compliance of predicted states.

    Args:
        states: [T, N, D] predicted state trajectories
        variable_names: names for each dimension

    Returns:
        Dict with overall and per-variable compliance rates and violation stats.
    """
    if variable_names is None:
        variable_names = VARIABLE_NAMES[: states.shape[-1]]

    total_points = states.shape[0] * states.shape[1]
    total_violations = 0
    per_variable = {}

    for d, name in enumerate(variable_names):
        if d >= states.shape[-1]:
            break

        bounds = PHYSIOLOGICAL_BOUNDS.get(name)
        if bounds is None:
            continue

        lo, hi, unit = bounds
        values = states[:, :, d]

        below = (values < lo).sum()
        above = (values > hi).sum()
        violations = int(below + above)
        total_violations += violations

        violation_rate = violations / max(total_points, 1)

        per_variable[name] = {
            "bounds": f"[{lo}, {hi}] {unit}",
            "violations": violations,
            "violation_rate": violation_rate,
            "compliant_rate": 1.0 - violation_rate,
            "min_observed": float(values.min()),
            "max_observed": float(values.max()),
            "mean_observed": float(values.mean()),
        }

    overall_violation_rate = total_violations / max(total_points * len(variable_names), 1)

    return {
        "overall_compliance_rate": 1.0 - overall_violation_rate,
        "overall_violation_rate": overall_violation_rate,
        "total_violations": total_violations,
        "total_evaluation_points": total_points * len(variable_names),
        "per_variable": per_variable,
    }
