"""
PK-Informed Biomarker Trajectory Generator

Generates realistic GFAP, UCH-L1, S100B, and NfL temporal profiles
conditioned on injury severity, using population pharmacokinetic (PK)
parameters with inter-individual variability (IIV).

PK parameters sourced from:
  - GFAP: TRACK-TBI, Cmax 20-24h, t½ 24-36h
  - UCH-L1: TRACK-TBI, Cmax 8-12h, t½ 7-9h
  - S100B: Scandinavian guidelines, Cmax 6h, t½ ~2h
  - NfL: Delayed rise, peak 10-30d, t½ weeks

Canonical thresholds (Abbott i-STAT / NINDS):
  GFAP >= 30 pg/mL  -> CT indicated
  UCH-L1 >= 360 pg/mL -> CT indicated
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    from ode_systems.biomarker_kinetics import GFAP_CT_THRESHOLD, UCHL1_CT_THRESHOLD
except ImportError:  # pragma: no cover - fallback for package-relative execution
    from ..ode_systems.biomarker_kinetics import GFAP_CT_THRESHOLD, UCHL1_CT_THRESHOLD

try:
    from synthetic.subgroup_profiles import SubgroupProfile, ADULT_MODERATE_SEVERE_TBI
except ImportError:  # pragma: no cover - fallback for package-relative execution
    from .subgroup_profiles import SubgroupProfile, ADULT_MODERATE_SEVERE_TBI


GFAP_HALF_LIFE_RANGE_HOURS = (24.0, 36.0)
UCHL1_HALF_LIFE_RANGE_HOURS = (7.0, 9.0)


class BiomarkerGenerator:
    """
    Generate synthetic biomarker kinetic profiles using PK-informed models
    with population variability.

    Model per analyte:
      C(t) = Cmax * (exp(-k_el * t) - exp(-k_abs * t)) / (1 - k_el/k_abs)

    Where k_abs and k_el are derived from tmax and t½ respectively,
    and Cmax is drawn from the profile's log-normal distribution.
    """

    def __init__(
        self,
        profile: SubgroupProfile | None = None,
        seed: int = 42,
        include_s100b: bool = False,
        include_nfl: bool = False,
    ):
        self.profile = profile or ADULT_MODERATE_SEVERE_TBI
        self.rng = np.random.default_rng(seed)
        self.include_s100b = include_s100b
        self.include_nfl = include_nfl

    def _sample_pk_params(self) -> dict[str, dict[str, float]]:
        """Sample patient-specific PK parameters with IIV (log-normal)."""
        bp = self.profile.biomarker

        # Log-normal sampling for Cmax (ensures positivity)
        def _lognormal(mean: float, std: float) -> float:
            if mean <= 0 or std <= 0:
                return max(0.1, mean)
            sigma2 = np.log(1 + (std / mean) ** 2)
            mu = np.log(mean) - sigma2 / 2
            return float(self.rng.lognormal(mu, np.sqrt(sigma2)))

        params = {
            "GFAP": {
                "cmax": _lognormal(bp.gfap_cmax_mean, bp.gfap_cmax_std),
                "tmax_hours": bp.gfap_tmax_hours * self.rng.uniform(0.8, 1.2),
                "half_life_hours": float(np.clip(
                    self.rng.normal(bp.gfap_half_life_hours, bp.gfap_half_life_std),
                    GFAP_HALF_LIFE_RANGE_HOURS[0],
                    GFAP_HALF_LIFE_RANGE_HOURS[1],
                )),
            },
            "UCH_L1": {
                "cmax": _lognormal(bp.uchl1_cmax_mean, bp.uchl1_cmax_std),
                "tmax_hours": bp.uchl1_tmax_hours * self.rng.uniform(0.8, 1.2),
                "half_life_hours": float(np.clip(
                    self.rng.normal(bp.uchl1_half_life_hours, bp.uchl1_half_life_std),
                    UCHL1_HALF_LIFE_RANGE_HOURS[0],
                    UCHL1_HALF_LIFE_RANGE_HOURS[1],
                )),
            },
        }

        if self.include_s100b:
            params["S100B"] = {
                "cmax": _lognormal(bp.s100b_cmax_mean, bp.s100b_cmax_std),
                "tmax_hours": bp.s100b_tmax_hours * self.rng.uniform(0.8, 1.2),
                "half_life_hours": max(0.5, bp.s100b_half_life_hours + self.rng.normal(0, 0.3)),
            }

        if self.include_nfl:
            params["NfL"] = {
                "cmax": _lognormal(bp.nfl_cmax_mean, bp.nfl_cmax_std),
                "tmax_hours": bp.nfl_tmax_days * 24 * self.rng.uniform(0.7, 1.3),
                "half_life_hours": bp.nfl_half_life_days * 24 * self.rng.uniform(0.8, 1.2),
            }

        return params

    @staticmethod
    def _bateman_function(
        t: np.ndarray,
        cmax: float,
        tmax: float,
        half_life: float,
    ) -> np.ndarray:
        """
        Bateman function (one-compartment oral PK model).

        C(t) = A * (exp(-k_el * t) - exp(-k_abs * t))

        Where k_el = ln(2)/t½, k_abs derived from tmax constraint.
        """
        k_el = np.log(2) / half_life

        # k_abs from tmax: tmax = ln(k_abs/k_el) / (k_abs - k_el)
        # Approximate: k_abs ≈ 1/tmax * (1 + k_el*tmax) for reasonable ranges
        if tmax > 0:
            k_abs = max(k_el * 1.5, np.log(2) / (tmax * 0.3))
        else:
            k_abs = k_el * 5.0

        # Ensure k_abs > k_el
        k_abs = max(k_abs, k_el * 1.1)

        # Normalization factor so peak = cmax
        if abs(k_abs - k_el) > 1e-8:
            t_peak = np.log(k_abs / k_el) / (k_abs - k_el)
            c_peak = np.exp(-k_el * t_peak) - np.exp(-k_abs * t_peak)
            if c_peak > 1e-10:
                A = cmax / c_peak
            else:
                A = cmax
        else:
            A = cmax

        conc = A * (np.exp(-k_el * t) - np.exp(-k_abs * t))
        return np.maximum(conc, 0.0)

    def _add_noise(self, trajectory: np.ndarray, cv: float = 0.15) -> np.ndarray:
        """Add proportional measurement noise (coefficient of variation)."""
        noise = self.rng.normal(1.0, cv, size=trajectory.shape)
        noisy = trajectory * noise
        return np.maximum(noisy, 0.0)

    def generate_patient(
        self,
        patient_id: str | None = None,
        times_hours: np.ndarray | None = None,
    ) -> dict:
        """
        Generate biomarker trajectories for a single patient.

        Args:
            patient_id: Patient identifier
            times_hours: Observation times. If None, generates irregular times.

        Returns dict with keys:
          patient_id, times, analytes (dict of arrays), pk_params,
          threshold_crossings, profile_name
        """
        if patient_id is None:
            patient_id = f"BM_{self.rng.integers(0, 99999):05d}"

        pk_params = self._sample_pk_params()

        # Generate observation times if not provided
        if times_hours is None:
            max_hours = self.profile.max_observation_hours
            density = self.profile.observation_density
            n_obs = max(3, int(density * max_hours))
            times_hours = np.sort(self.rng.uniform(0, max_hours, size=n_obs))
            # Always include t=0
            times_hours = np.concatenate([[0.0], times_hours])
            times_hours = np.unique(times_hours)

        # Generate trajectories
        analytes = {}
        for name, params in pk_params.items():
            clean = self._bateman_function(
                times_hours, params["cmax"], params["tmax_hours"], params["half_life_hours"]
            )
            analytes[name] = self._add_noise(clean)

        # Evaluate threshold crossings
        threshold_crossings = {}
        if "GFAP" in analytes:
            gfap_max = float(np.max(analytes["GFAP"]))
            threshold_crossings["gfap_max_pg_ml"] = gfap_max
            threshold_crossings["gfap_ct_positive"] = gfap_max >= GFAP_CT_THRESHOLD
        if "UCH_L1" in analytes:
            uchl1_max = float(np.max(analytes["UCH_L1"]))
            threshold_crossings["uchl1_max_pg_ml"] = uchl1_max
            threshold_crossings["uchl1_ct_positive"] = uchl1_max >= UCHL1_CT_THRESHOLD
        if "GFAP" in analytes and "UCH_L1" in analytes:
            threshold_crossings["ct_indicated"] = (
                threshold_crossings.get("gfap_ct_positive", False)
                or threshold_crossings.get("uchl1_ct_positive", False)
            )
            threshold_crossings["biomarker_negative"] = (
                not threshold_crossings.get("gfap_ct_positive", True)
                and not threshold_crossings.get("uchl1_ct_positive", True)
            )

        return {
            "patient_id": patient_id,
            "times": times_hours,
            "analytes": analytes,
            "pk_params": pk_params,
            "threshold_crossings": threshold_crossings,
            "profile_name": self.profile.name,
        }

    def generate_cohort(
        self,
        n_patients: int = 100,
        id_prefix: str = "BM",
    ) -> pd.DataFrame:
        """
        Generate a cohort of biomarker trajectories as a long-format DataFrame.

        Columns: patient_id, time_hours, GFAP, UCH_L1, [S100B], [NfL],
                 gfap_ct_positive, uchl1_ct_positive, ct_indicated, profile
        """
        rows = []
        for i in range(n_patients):
            pid = f"{id_prefix}_{i:04d}"
            patient = self.generate_patient(patient_id=pid)

            for t_idx in range(len(patient["times"])):
                row = {
                    "patient_id": pid,
                    "time_hours": patient["times"][t_idx],
                }
                for analyte_name, values in patient["analytes"].items():
                    row[analyte_name] = values[t_idx]

                row["gfap_ct_positive"] = patient["threshold_crossings"].get("gfap_ct_positive", False)
                row["uchl1_ct_positive"] = patient["threshold_crossings"].get("uchl1_ct_positive", False)
                row["ct_indicated"] = patient["threshold_crossings"].get("ct_indicated", False)
                row["profile"] = patient["profile_name"]
                rows.append(row)

        return pd.DataFrame(rows)
