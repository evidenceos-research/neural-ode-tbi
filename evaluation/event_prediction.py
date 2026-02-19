"""
Event prediction evaluation for clinically meaningful events.

Evaluates the model's ability to predict:
  - ICP crisis (ICP > 22 mmHg)
  - Biomarker threshold crossings (GFAP >= 30, UCH-L1 >= 360)
  - CPP critical events (CPP < 60 mmHg)
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


ICP_CRISIS_THRESHOLD = 22.0
GFAP_CT_THRESHOLD = 30.0
UCHL1_CT_THRESHOLD = 360.0
CPP_CRITICAL_THRESHOLD = 60.0


def _binary_event_metrics(
    pred_values: np.ndarray,
    true_values: np.ndarray,
    threshold: float,
    direction: str = "above",
) -> dict:
    """Compute binary classification metrics for threshold-crossing events."""
    if direction == "above":
        pred_events = (pred_values >= threshold).astype(int)
        true_events = (true_values >= threshold).astype(int)
    else:
        pred_events = (pred_values < threshold).astype(int)
        true_events = (true_values < threshold).astype(int)

    if true_events.sum() == 0 and pred_events.sum() == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "auc": 1.0,
            "n_true_events": 0,
            "n_pred_events": 0,
        }

    if len(np.unique(true_events)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(true_events, pred_values))

    return {
        "precision": float(precision_score(true_events, pred_events, zero_division=0)),
        "recall": float(recall_score(true_events, pred_events, zero_division=0)),
        "f1": float(f1_score(true_events, pred_events, zero_division=0)),
        "auc": auc,
        "n_true_events": int(true_events.sum()),
        "n_pred_events": int(pred_events.sum()),
    }


def evaluate_event_prediction(
    predictions: np.ndarray,
    targets: np.ndarray,
    variable_indices: dict[str, int] | None = None,
) -> dict:
    """
    Evaluate event prediction for clinically meaningful thresholds.

    Args:
        predictions: [T, N, D] predicted trajectories
        targets: [T, N, D] true trajectories
        variable_indices: mapping of variable name to dimension index
            Default: {"ICP": 0, "GFAP": 5, "UCH_L1": 6}

    Returns:
        Dict with per-event-type metrics.
    """
    if variable_indices is None:
        variable_indices = {"ICP": 0, "GFAP": 5, "UCH_L1": 6}

    results = {}

    if "ICP" in variable_indices:
        idx = variable_indices["ICP"]
        results["icp_crisis"] = _binary_event_metrics(
            predictions[:, :, idx].flatten(),
            targets[:, :, idx].flatten(),
            ICP_CRISIS_THRESHOLD,
            direction="above",
        )

    if "GFAP" in variable_indices:
        idx = variable_indices["GFAP"]
        results["gfap_positive"] = _binary_event_metrics(
            predictions[:, :, idx].flatten(),
            targets[:, :, idx].flatten(),
            GFAP_CT_THRESHOLD,
            direction="above",
        )

    if "UCH_L1" in variable_indices:
        idx = variable_indices["UCH_L1"]
        results["uchl1_positive"] = _binary_event_metrics(
            predictions[:, :, idx].flatten(),
            targets[:, :, idx].flatten(),
            UCHL1_CT_THRESHOLD,
            direction="above",
        )

    return results
