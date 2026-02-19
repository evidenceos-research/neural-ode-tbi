"""
Full Synthetic Cohort Generator

Assembles complete synthetic TBI patient records by combining:
  - Physiological trajectories (PhysiologyGenerator)
  - Biomarker kinetics (BiomarkerGenerator)
  - Demographics (from SubgroupProfile)
  - Injury characteristics (GCS, mechanism, Marshall grade)
  - Outcomes (GOSE, mortality, ICU admission)

Outputs a unified DataFrame suitable for Neural ODE training,
CCR testing, and evaluation benchmarking.
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    from synthetic.physiology_generator import PhysiologyGenerator
    from synthetic.biomarker_generator import BiomarkerGenerator
    from synthetic.subgroup_profiles import (
        SubgroupProfile,
        ADULT_MODERATE_SEVERE_TBI,
        PROFILE_REGISTRY,
    )
except ImportError:  # pragma: no cover - fallback for package-relative execution
    from .physiology_generator import PhysiologyGenerator
    from .biomarker_generator import BiomarkerGenerator
    from .subgroup_profiles import (
        SubgroupProfile,
        ADULT_MODERATE_SEVERE_TBI,
        PROFILE_REGISTRY,
    )


class CohortGenerator:
    """
    Generate complete synthetic TBI cohorts with demographics,
    physiology, biomarkers, and outcomes.
    """

    def __init__(
        self,
        profiles: list[SubgroupProfile] | None = None,
        seed: int = 42,
        noise_scale: float = 1.0,
        include_s100b: bool = False,
        include_nfl: bool = False,
    ):
        self.profiles = profiles or [ADULT_MODERATE_SEVERE_TBI]
        self.seed = seed
        self.noise_scale = noise_scale
        self.include_s100b = include_s100b
        self.include_nfl = include_nfl
        self.rng = np.random.default_rng(seed)

    def _sample_demographics(
        self, profile: SubgroupProfile, rng: np.random.Generator
    ) -> dict:
        """Sample patient demographics from profile distributions."""
        demo = profile.demographics
        age = float(np.clip(
            rng.normal(demo.age_mean, demo.age_std),
            demo.age_min, demo.age_max,
        ))
        sex = "male" if rng.random() < demo.male_fraction else "female"

        mechanisms = list(demo.mechanism_weights.keys())
        weights = list(demo.mechanism_weights.values())
        mechanism = rng.choice(mechanisms, p=weights)

        anticoagulant = bool(rng.random() < demo.anticoagulant_rate)
        intoxication = bool(rng.random() < demo.intoxication_rate)

        return {
            "age": round(age, 1),
            "sex": sex,
            "mechanism": mechanism,
            "anticoagulant": anticoagulant,
            "intoxication": intoxication,
        }

    def _sample_injury(
        self, profile: SubgroupProfile, rng: np.random.Generator
    ) -> dict:
        """Sample injury characteristics."""
        gcs = int(rng.integers(profile.gcs_range[0], profile.gcs_range[1] + 1))
        marshall = int(rng.integers(profile.marshall_range[0], profile.marshall_range[1] + 1))

        # Pupil reactivity (correlated with GCS)
        if gcs <= 5:
            pupil_probs = [0.3, 0.3, 0.4]  # both_reactive, one, both_unreactive
        elif gcs <= 8:
            pupil_probs = [0.6, 0.25, 0.15]
        elif gcs <= 12:
            pupil_probs = [0.85, 0.10, 0.05]
        else:
            pupil_probs = [0.95, 0.04, 0.01]

        pupil = rng.choice(
            ["both_reactive", "one_unreactive", "both_unreactive"],
            p=pupil_probs,
        )

        return {
            "gcs": gcs,
            "marshall_grade": marshall,
            "pupil_reactivity": pupil,
        }

    def _sample_outcome(
        self, profile: SubgroupProfile, injury: dict, rng: np.random.Generator
    ) -> dict:
        """Sample outcomes conditioned on injury severity."""
        out = profile.outcomes

        # Mortality (higher for lower GCS)
        gcs = injury["gcs"]
        mortality_modifier = 1.0
        if gcs <= 5:
            mortality_modifier = 2.5
        elif gcs <= 8:
            mortality_modifier = 1.5
        elif gcs <= 12:
            mortality_modifier = 1.0
        else:
            mortality_modifier = 0.3

        mortality = bool(rng.random() < min(0.95, out.mortality_rate * mortality_modifier))

        if mortality:
            gose = 1
        else:
            gose = int(rng.choice(range(2, 9), p=np.array(out.gose_weights[1:]) / sum(out.gose_weights[1:])))

        icu = bool(rng.random() < out.icu_admission_rate)
        neurosurgery = bool(rng.random() < out.neurosurgery_rate)

        return {
            "gose_6mo": gose,
            "mortality": mortality,
            "icu_admission": icu,
            "neurosurgery": neurosurgery,
        }

    def generate(
        self,
        n_patients_per_profile: int | dict[str, int] | None = None,
        total_patients: int | None = None,
    ) -> dict:
        """
        Generate a complete synthetic cohort.

        Args:
            n_patients_per_profile: Number of patients per profile.
                Can be int (same for all) or dict mapping profile name to count.
            total_patients: If set, distributes evenly across profiles.

        Returns dict with keys:
          - trajectories: pd.DataFrame (long-format time series)
          - demographics: pd.DataFrame (one row per patient)
          - summary: dict with cohort statistics
        """
        if total_patients is not None:
            n_per = total_patients // len(self.profiles)
            counts = {p.name: n_per for p in self.profiles}
            # Distribute remainder
            remainder = total_patients - n_per * len(self.profiles)
            for i, p in enumerate(self.profiles):
                if i < remainder:
                    counts[p.name] += 1
        elif isinstance(n_patients_per_profile, int):
            counts = {p.name: n_patients_per_profile for p in self.profiles}
        elif isinstance(n_patients_per_profile, dict):
            counts = n_patients_per_profile
        else:
            counts = {p.name: 50 for p in self.profiles}

        all_traj_rows = []
        all_demo_rows = []
        patient_counter = 0

        for profile in self.profiles:
            n = counts.get(profile.name, 50)
            profile_seed = self.seed + hash(profile.name) % 10000

            phys_gen = PhysiologyGenerator(
                profile=profile, seed=profile_seed, noise_scale=self.noise_scale,
            )
            bm_gen = BiomarkerGenerator(
                profile=profile, seed=profile_seed + 1,
                include_s100b=self.include_s100b, include_nfl=self.include_nfl,
            )
            rng = np.random.default_rng(profile_seed + 2)

            for i in range(n):
                pid = f"COHORT_{patient_counter:05d}"
                patient_counter += 1

                # Demographics & injury
                demographics = self._sample_demographics(profile, rng)
                injury = self._sample_injury(profile, rng)
                outcome = self._sample_outcome(profile, injury, rng)

                # Physiology trajectory
                phys = phys_gen.generate_patient(patient_id=pid)

                # Biomarker trajectory (aligned to same times)
                bm = bm_gen.generate_patient(
                    patient_id=pid, times_hours=phys["times"]
                )

                # Merge into trajectory rows
                for t_idx in range(len(phys["times"])):
                    row = {
                        "patient_id": pid,
                        "time_hours": phys["times"][t_idx],
                        "ICP": phys["observations"][t_idx, 0],
                        "V_csf": phys["observations"][t_idx, 1],
                        "V_cbv": phys["observations"][t_idx, 2],
                        "CVR": phys["observations"][t_idx, 3],
                        "CBF": phys["observations"][t_idx, 4],
                        "MAP": phys["map_values"][t_idx],
                        "CPP": phys["cpp_values"][t_idx],
                    }
                    for analyte_name, values in bm["analytes"].items():
                        row[analyte_name] = values[t_idx]

                    # Masks
                    for j, vname in enumerate(["ICP", "V_csf", "V_cbv", "CVR", "CBF"]):
                        row[f"mask_{vname}"] = phys["mask"][t_idx, j]

                    all_traj_rows.append(row)

                # Demographics row
                demo_row = {
                    "patient_id": pid,
                    "profile": profile.name,
                    **demographics,
                    **injury,
                    **outcome,
                    **bm["threshold_crossings"],
                    "imaging_available": profile.imaging_available,
                    "biomarker_available": profile.biomarker_available,
                    "n_observations": len(phys["times"]),
                    "max_time_hours": float(phys["times"][-1]),
                }
                all_demo_rows.append(demo_row)

        trajectories = pd.DataFrame(all_traj_rows)
        demographics = pd.DataFrame(all_demo_rows)

        summary = {
            "total_patients": len(demographics),
            "total_observations": len(trajectories),
            "profiles": {p.name: counts.get(p.name, 0) for p in self.profiles},
            "gcs_distribution": demographics["gcs"].describe().to_dict() if "gcs" in demographics else {},
            "mortality_rate": float(demographics["mortality"].mean()) if "mortality" in demographics else 0.0,
            "ct_indicated_rate": float(demographics["ct_indicated"].mean()) if "ct_indicated" in demographics else 0.0,
        }

        return {
            "trajectories": trajectories,
            "demographics": demographics,
            "summary": summary,
        }

    def save(
        self,
        output_dir: str,
        cohort: dict | None = None,
        **generate_kwargs,
    ) -> dict[str, str]:
        """
        Generate and save cohort to CSV files.

        Returns dict mapping output type to file path.
        """
        from pathlib import Path

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if cohort is None:
            cohort = self.generate(**generate_kwargs)

        traj_path = out_path / "trajectories.csv"
        demo_path = out_path / "demographics.csv"
        summary_path = out_path / "summary.json"

        cohort["trajectories"].to_csv(traj_path, index=False)
        cohort["demographics"].to_csv(demo_path, index=False)

        import json
        with open(summary_path, "w") as f:
            json.dump(cohort["summary"], f, indent=2, default=str)

        return {
            "trajectories": str(traj_path),
            "demographics": str(demo_path),
            "summary": str(summary_path),
        }
