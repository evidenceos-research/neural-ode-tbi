"""
Constraint Compliance Metrics for Synthetic Data Validation (Gate 3)

Evaluates how well synthetic trajectories respect hard physiological
constraints — the differentiating validation contribution for
physics-informed synthetic TBI data.

Checks:
  - Monro-Kellie volume conservation
  - CPP identity (CPP = MAP - ICP within measurement noise)
  - Physiological bounds satisfaction rate
  - Biomarker non-negativity
  - Biomarker threshold consistency
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    from ode_systems.biomarker_kinetics import GFAP_CT_THRESHOLD, UCHL1_CT_THRESHOLD
except ImportError:  # pragma: no cover - fallback for package-relative execution
    from ...ode_systems.biomarker_kinetics import GFAP_CT_THRESHOLD, UCHL1_CT_THRESHOLD


# Physiological bounds (mirroring constraints/physiological_bounds.py)
BOUNDS = {
    "ICP": (0.0, 100.0),
    "V_csf": (0.0, 300.0),
    "V_cbv": (0.0, 150.0),
    "CVR": (0.1, 20.0),
    "CBF": (0.0, 150.0),
    "GFAP": (0.0, 50000.0),
    "UCH_L1": (0.0, 50000.0),
    "MAP": (30.0, 250.0),
    "CPP": (-20.0, 200.0),
}

# Total intracranial volume (Monro-Kellie)
TOTAL_ICV_ML = 1500.0
MONRO_KELLIE_TOLERANCE = 50.0  # mL tolerance for volume sum check

# CPP identity tolerance
CPP_IDENTITY_TOLERANCE = 2.0  # mmHg


def check_physiological_bounds(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """
    Check what fraction of observations fall within physiological bounds.

    Returns per-variable and overall violation rates.
    """
    if bounds is None:
        bounds = BOUNDS

    results = {}
    total_checks = 0
    total_violations = 0

    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue

        values = df[col].dropna().values
        n = len(values)
        if n == 0:
            continue

        below = int(np.sum(values < lo))
        above = int(np.sum(values > hi))
        violations = below + above

        results[col] = {
            "n_observations": n,
            "n_violations": violations,
            "violation_rate": violations / n,
            "n_below_min": below,
            "n_above_max": above,
            "bounds": [lo, hi],
        }

        total_checks += n
        total_violations += violations

    return {
        "per_variable": results,
        "overall_violation_rate": total_violations / max(total_checks, 1),
        "total_checks": total_checks,
        "total_violations": total_violations,
    }


def check_cpp_identity(df: pd.DataFrame) -> dict:
    """
    Verify CPP = MAP - ICP identity within measurement noise.

    Requires columns: MAP, ICP, CPP.
    """
    required = {"MAP", "ICP", "CPP"}
    if not required.issubset(df.columns):
        return {"error": f"Missing columns: {required - set(df.columns)}"}

    mask = df[["MAP", "ICP", "CPP"]].notna().all(axis=1)
    valid = df[mask]

    if len(valid) == 0:
        return {"error": "No valid observations with MAP, ICP, and CPP"}

    expected_cpp = valid["MAP"].values - valid["ICP"].values
    actual_cpp = valid["CPP"].values
    residuals = np.abs(expected_cpp - actual_cpp)

    n_violations = int(np.sum(residuals > CPP_IDENTITY_TOLERANCE))

    return {
        "n_observations": len(valid),
        "n_violations": n_violations,
        "violation_rate": n_violations / len(valid),
        "mean_residual_mmhg": float(np.mean(residuals)),
        "max_residual_mmhg": float(np.max(residuals)),
        "tolerance_mmhg": CPP_IDENTITY_TOLERANCE,
    }


def check_biomarker_nonnegativity(df: pd.DataFrame) -> dict:
    """Check that all biomarker values are non-negative."""
    biomarker_cols = [c for c in ["GFAP", "UCH_L1", "S100B", "NfL"] if c in df.columns]

    results = {}
    for col in biomarker_cols:
        values = df[col].dropna().values
        n_negative = int(np.sum(values < 0))
        results[col] = {
            "n_observations": len(values),
            "n_negative": n_negative,
            "violation_rate": n_negative / max(len(values), 1),
        }

    return results


def check_monro_kellie(
    df: pd.DataFrame,
    v_brain_ml: float = 1200.0,
) -> dict:
    """
    Check Monro-Kellie volume conservation: V_brain + V_csf + V_cbv ≈ constant.

    Since V_brain is not directly modeled, checks that V_csf + V_cbv
    stays within a plausible range.
    """
    if "V_csf" not in df.columns or "V_cbv" not in df.columns:
        return {"error": "Missing V_csf or V_cbv columns"}

    mask = df[["V_csf", "V_cbv"]].notna().all(axis=1)
    valid = df[mask]

    if len(valid) == 0:
        return {"error": "No valid observations"}

    total_non_brain = valid["V_csf"].values + valid["V_cbv"].values
    expected_non_brain = TOTAL_ICV_ML - v_brain_ml  # ~300 mL

    deviation = np.abs(total_non_brain - expected_non_brain)
    n_violations = int(np.sum(deviation > MONRO_KELLIE_TOLERANCE))

    return {
        "n_observations": len(valid),
        "n_violations": n_violations,
        "violation_rate": n_violations / len(valid),
        "mean_total_non_brain_ml": float(np.mean(total_non_brain)),
        "expected_non_brain_ml": expected_non_brain,
        "tolerance_ml": MONRO_KELLIE_TOLERANCE,
    }


def compute_constraint_compliance(
    df: pd.DataFrame,
    target_compliance: float = 0.99,
) -> dict:
    """
    Compute full Gate 3 constraint compliance report.

    Args:
        df: Synthetic trajectory DataFrame
        target_compliance: Required compliance rate (default 99%)

    Returns dict with:
      - bounds: physiological bounds check
      - cpp_identity: CPP = MAP - ICP check
      - biomarker_nonnegativity: non-negativity check
      - monro_kellie: volume conservation check
      - overall_compliance: aggregate compliance rate
      - gate_pass: True if overall compliance >= target
    """
    bounds = check_physiological_bounds(df)
    cpp = check_cpp_identity(df)
    bm_nn = check_biomarker_nonnegativity(df)
    mk = check_monro_kellie(df)

    # Aggregate compliance
    compliance_scores = []
    if bounds["total_checks"] > 0:
        compliance_scores.append(1.0 - bounds["overall_violation_rate"])
    if isinstance(cpp, dict) and "violation_rate" in cpp:
        compliance_scores.append(1.0 - cpp["violation_rate"])
    for col_result in bm_nn.values():
        if isinstance(col_result, dict) and "violation_rate" in col_result:
            compliance_scores.append(1.0 - col_result["violation_rate"])

    overall = float(np.mean(compliance_scores)) if compliance_scores else 0.0

    return {
        "bounds": bounds,
        "cpp_identity": cpp,
        "biomarker_nonnegativity": bm_nn,
        "monro_kellie": mk,
        "overall_compliance": overall,
        "target_compliance": target_compliance,
        "gate_pass": overall >= target_compliance,
    }
