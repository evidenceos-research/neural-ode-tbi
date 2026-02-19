"""
Trajectory-level evaluation metrics for Neural ODE predictions.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_trajectory_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray | None = None,
    variable_names: list[str] | None = None,
) -> dict:
    """
    Compute per-variable and aggregate trajectory metrics.

    Args:
        predictions: [T, N, D] predicted values
        targets: [T, N, D] ground truth values
        mask: [T, N, D] binary mask (1 = observed)
        variable_names: names for each variable dimension

    Returns:
        Dict with aggregate and per-variable metrics.
    """
    if variable_names is None:
        variable_names = [f"var_{i}" for i in range(predictions.shape[-1])]

    results = {"aggregate": {}, "per_variable": {}}

    # Flatten for aggregate metrics
    if mask is not None:
        valid = mask.astype(bool)
        pred_flat = predictions[valid]
        tgt_flat = targets[valid]
    else:
        pred_flat = predictions.reshape(-1)
        tgt_flat = targets.reshape(-1)

    if len(pred_flat) > 0:
        results["aggregate"] = {
            "mse": float(mean_squared_error(tgt_flat, pred_flat)),
            "rmse": float(np.sqrt(mean_squared_error(tgt_flat, pred_flat))),
            "mae": float(mean_absolute_error(tgt_flat, pred_flat)),
            "r2": float(r2_score(tgt_flat, pred_flat)) if len(pred_flat) > 1 else 0.0,
            "n_observations": int(len(pred_flat)),
        }

    # Per-variable metrics
    for d, name in enumerate(variable_names):
        if d >= predictions.shape[-1]:
            break

        if mask is not None:
            valid_d = mask[:, :, d].astype(bool)
            p = predictions[:, :, d][valid_d]
            t = targets[:, :, d][valid_d]
        else:
            p = predictions[:, :, d].reshape(-1)
            t = targets[:, :, d].reshape(-1)

        if len(p) == 0:
            continue

        results["per_variable"][name] = {
            "mse": float(mean_squared_error(t, p)),
            "rmse": float(np.sqrt(mean_squared_error(t, p))),
            "mae": float(mean_absolute_error(t, p)),
            "r2": float(r2_score(t, p)) if len(p) > 1 else 0.0,
            "n_observations": int(len(p)),
        }

    return results
