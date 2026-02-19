"""
Statistical Fidelity Metrics for Synthetic Data Validation (Gate 1)

Evaluates how well synthetic data preserves the statistical properties
of real (or reference) data:
  - Column-wise KS test (marginal distributions)
  - Correlation matrix preservation (Frobenius norm)
  - Summary statistics comparison
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


def ks_test_per_column(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    Perform two-sample KS test for each numeric column.

    Args:
        real: Reference (real) data
        synthetic: Synthetic data
        columns: Columns to test. If None, uses all shared numeric columns.
        alpha: Significance level

    Returns dict with per-column KS statistic, p-value, and pass/fail.
    """
    if columns is None:
        shared = set(real.select_dtypes(include=[np.number]).columns) & \
                 set(synthetic.select_dtypes(include=[np.number]).columns)
        columns = sorted(shared)

    results = {}
    for col in columns:
        r = real[col].dropna().values
        s = synthetic[col].dropna().values
        if len(r) < 2 or len(s) < 2:
            results[col] = {"ks_statistic": None, "p_value": None, "pass": None}
            continue

        ks_stat, p_val = stats.ks_2samp(r, s)
        results[col] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_val),
            "pass": bool(p_val > alpha),
        }

    return results


def correlation_preservation(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict:
    """
    Compare correlation matrices between real and synthetic data.

    Returns Frobenius norm of the difference and per-pair max deviation.
    """
    if columns is None:
        shared = set(real.select_dtypes(include=[np.number]).columns) & \
                 set(synthetic.select_dtypes(include=[np.number]).columns)
        columns = sorted(shared)

    real_corr = real[columns].corr().values
    synth_corr = synthetic[columns].corr().values

    # Handle NaN in correlation matrices
    real_corr = np.nan_to_num(real_corr, nan=0.0)
    synth_corr = np.nan_to_num(synth_corr, nan=0.0)

    diff = real_corr - synth_corr
    frobenius = float(np.linalg.norm(diff, "fro"))
    max_deviation = float(np.max(np.abs(diff)))

    return {
        "frobenius_norm": frobenius,
        "max_pair_deviation": max_deviation,
        "n_variables": len(columns),
        "columns": columns,
    }


def summary_statistics_comparison(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict:
    """
    Compare mean, std, median, min, max between real and synthetic.
    """
    if columns is None:
        shared = set(real.select_dtypes(include=[np.number]).columns) & \
                 set(synthetic.select_dtypes(include=[np.number]).columns)
        columns = sorted(shared)

    results = {}
    for col in columns:
        r = real[col].dropna()
        s = synthetic[col].dropna()
        results[col] = {
            "real_mean": float(r.mean()) if len(r) > 0 else None,
            "synth_mean": float(s.mean()) if len(s) > 0 else None,
            "real_std": float(r.std()) if len(r) > 0 else None,
            "synth_std": float(s.std()) if len(s) > 0 else None,
            "real_median": float(r.median()) if len(r) > 0 else None,
            "synth_median": float(s.median()) if len(s) > 0 else None,
            "mean_relative_error": (
                abs(float(r.mean()) - float(s.mean())) / max(abs(float(r.mean())), 1e-8)
                if len(r) > 0 and len(s) > 0 else None
            ),
        }

    return results


def compute_fidelity_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    Compute full Gate 1 fidelity report.

    Returns dict with:
      - ks_tests: per-column KS test results
      - correlation: correlation preservation metrics
      - summary_stats: per-column summary statistics comparison
      - overall_pass: True if all KS tests pass and correlation Frobenius < threshold
    """
    ks = ks_test_per_column(real, synthetic, columns, alpha)
    corr = correlation_preservation(real, synthetic, columns)
    summary = summary_statistics_comparison(real, synthetic, columns)

    ks_pass_rate = sum(1 for v in ks.values() if v.get("pass")) / max(len(ks), 1)
    corr_threshold = 2.0 * corr["n_variables"]  # scale-dependent threshold

    overall_pass = ks_pass_rate >= 0.8 and corr["frobenius_norm"] < corr_threshold

    return {
        "ks_tests": ks,
        "ks_pass_rate": ks_pass_rate,
        "correlation": corr,
        "summary_stats": summary,
        "overall_pass": overall_pass,
    }
