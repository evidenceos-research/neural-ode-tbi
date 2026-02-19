"""
Subgroup Profiles for Conditional Synthetic Data Generation

Defines injury severity, demographic, and deployment context profiles
that parameterize the synthetic generators. Each profile specifies
distributions over initial conditions, PK parameters, and patient
characteristics.

Profiles are designed to be loaded from YAML configs or used directly.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class PhysiologyProfile:
    """Distribution parameters for physiological initial conditions."""
    icp_mean: float = 12.0
    icp_std: float = 3.0
    map_mean: float = 85.0
    map_std: float = 10.0
    v_csf_mean: float = 120.0
    v_csf_std: float = 20.0
    v_cbv_mean: float = 50.0
    v_cbv_std: float = 8.0
    cvr_mean: float = 1.6
    cvr_std: float = 0.3
    cbf_mean: float = 50.0
    cbf_std: float = 10.0


@dataclass
class BiomarkerProfile:
    """Distribution parameters for biomarker PK and initial conditions."""
    # GFAP PK (population parameters)
    gfap_cmax_mean: float = 100.0       # pg/mL at peak
    gfap_cmax_std: float = 80.0
    gfap_tmax_hours: float = 20.0       # time to peak
    gfap_half_life_hours: float = 30.0  # canonical range 24-36h
    gfap_half_life_std: float = 3.0

    # UCH-L1 PK
    uchl1_cmax_mean: float = 400.0
    uchl1_cmax_std: float = 250.0
    uchl1_tmax_hours: float = 8.0
    uchl1_half_life_hours: float = 8.0
    uchl1_half_life_std: float = 1.0

    # S100B PK (optional, for extended models)
    s100b_cmax_mean: float = 0.5        # μg/L
    s100b_cmax_std: float = 0.3
    s100b_tmax_hours: float = 3.0
    s100b_half_life_hours: float = 1.5

    # NfL PK (slow rise, long persistence)
    nfl_cmax_mean: float = 50.0         # pg/mL
    nfl_cmax_std: float = 30.0
    nfl_tmax_days: float = 14.0
    nfl_half_life_days: float = 42.0    # ~6 weeks


@dataclass
class DemographicProfile:
    """Distribution parameters for patient demographics."""
    age_mean: float = 45.0
    age_std: float = 18.0
    age_min: float = 18.0
    age_max: float = 90.0
    male_fraction: float = 0.65
    mechanism_weights: dict[str, float] = field(default_factory=lambda: {
        "fall": 0.35,
        "rta": 0.30,
        "assault": 0.10,
        "sport": 0.10,
        "blast": 0.02,
        "other": 0.13,
    })
    anticoagulant_rate: float = 0.08
    intoxication_rate: float = 0.20


@dataclass
class OutcomeProfile:
    """Distribution parameters for outcomes."""
    # GOSE distribution weights (1-8)
    gose_weights: list[float] = field(default_factory=lambda: [
        0.05, 0.05, 0.10, 0.15, 0.15, 0.20, 0.15, 0.15
    ])
    mortality_rate: float = 0.05
    icu_admission_rate: float = 0.30
    neurosurgery_rate: float = 0.10


@dataclass
class SubgroupProfile:
    """Complete profile for a patient subgroup."""
    name: str
    description: str
    gcs_range: tuple[int, int] = (3, 15)
    marshall_range: tuple[int, int] = (1, 6)
    physiology: PhysiologyProfile = field(default_factory=PhysiologyProfile)
    biomarker: BiomarkerProfile = field(default_factory=BiomarkerProfile)
    demographics: DemographicProfile = field(default_factory=DemographicProfile)
    outcomes: OutcomeProfile = field(default_factory=OutcomeProfile)
    imaging_available: bool = True
    biomarker_available: bool = True
    max_observation_hours: float = 48.0
    observation_density: float = 1.0  # observations per hour (mean)


# =============================================================================
# Pre-defined profiles
# =============================================================================

ADULT_MILD_TBI = SubgroupProfile(
    name="adult_mild_tbi",
    description="Adult mild TBI (GCS 13-15), typical ED presentation",
    gcs_range=(13, 15),
    marshall_range=(1, 2),
    physiology=PhysiologyProfile(icp_mean=10.0, icp_std=2.0, map_mean=90.0),
    biomarker=BiomarkerProfile(
        gfap_cmax_mean=25.0, gfap_cmax_std=20.0,
        uchl1_cmax_mean=200.0, uchl1_cmax_std=150.0,
    ),
    demographics=DemographicProfile(age_mean=38.0, age_std=15.0),
    outcomes=OutcomeProfile(
        gose_weights=[0.01, 0.01, 0.03, 0.05, 0.10, 0.20, 0.25, 0.35],
        mortality_rate=0.005, icu_admission_rate=0.05, neurosurgery_rate=0.01,
    ),
    max_observation_hours=24.0,
    observation_density=0.5,
)

ADULT_MODERATE_SEVERE_TBI = SubgroupProfile(
    name="adult_moderate_severe_tbi",
    description="Adult moderate-severe TBI (GCS 3-12), ICU admission",
    gcs_range=(3, 12),
    marshall_range=(2, 6),
    physiology=PhysiologyProfile(
        icp_mean=22.0, icp_std=8.0, map_mean=80.0, map_std=12.0,
        cbf_mean=35.0, cbf_std=12.0,
    ),
    biomarker=BiomarkerProfile(
        gfap_cmax_mean=500.0, gfap_cmax_std=400.0,
        uchl1_cmax_mean=1500.0, uchl1_cmax_std=1000.0,
        nfl_cmax_mean=200.0, nfl_cmax_std=150.0,
    ),
    demographics=DemographicProfile(
        age_mean=42.0, age_std=20.0, intoxication_rate=0.30,
    ),
    outcomes=OutcomeProfile(
        gose_weights=[0.10, 0.10, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10],
        mortality_rate=0.15, icu_admission_rate=0.90, neurosurgery_rate=0.30,
    ),
    max_observation_hours=168.0,  # 7 days
    observation_density=2.0,
)

PEDIATRIC_TBI = SubgroupProfile(
    name="pediatric_tbi",
    description="Pediatric TBI (age 2-17), adjusted physiology",
    gcs_range=(3, 15),
    marshall_range=(1, 6),
    physiology=PhysiologyProfile(
        icp_mean=8.0, icp_std=3.0, map_mean=70.0, map_std=10.0,
        v_csf_mean=80.0, v_csf_std=15.0, v_cbv_mean=35.0, v_cbv_std=6.0,
        cbf_mean=60.0, cbf_std=12.0,
    ),
    biomarker=BiomarkerProfile(
        gfap_cmax_mean=80.0, gfap_cmax_std=60.0,
        uchl1_cmax_mean=350.0, uchl1_cmax_std=200.0,
    ),
    demographics=DemographicProfile(
        age_mean=10.0, age_std=4.0, age_min=2.0, age_max=17.0,
        male_fraction=0.60,
        mechanism_weights={"fall": 0.40, "rta": 0.15, "assault": 0.05,
                           "sport": 0.25, "blast": 0.0, "other": 0.15},
        anticoagulant_rate=0.0, intoxication_rate=0.02,
    ),
    outcomes=OutcomeProfile(
        gose_weights=[0.02, 0.03, 0.05, 0.10, 0.10, 0.20, 0.25, 0.25],
        mortality_rate=0.03, icu_admission_rate=0.20, neurosurgery_rate=0.08,
    ),
    max_observation_hours=72.0,
    observation_density=1.5,
)

LMIC_CONTEXT = SubgroupProfile(
    name="lmic_context",
    description="LMIC setting — limited imaging, biomarker-primary triage",
    gcs_range=(3, 15),
    marshall_range=(1, 6),
    physiology=PhysiologyProfile(icp_mean=15.0, icp_std=6.0),
    biomarker=BiomarkerProfile(
        gfap_cmax_mean=150.0, gfap_cmax_std=120.0,
        uchl1_cmax_mean=500.0, uchl1_cmax_std=350.0,
    ),
    demographics=DemographicProfile(
        age_mean=32.0, age_std=12.0,
        mechanism_weights={"fall": 0.20, "rta": 0.50, "assault": 0.15,
                           "sport": 0.03, "blast": 0.02, "other": 0.10},
        anticoagulant_rate=0.02, intoxication_rate=0.15,
    ),
    outcomes=OutcomeProfile(
        gose_weights=[0.12, 0.10, 0.12, 0.15, 0.15, 0.15, 0.11, 0.10],
        mortality_rate=0.20, icu_admission_rate=0.40, neurosurgery_rate=0.15,
    ),
    imaging_available=False,
    max_observation_hours=48.0,
    observation_density=0.3,
)

MILITARY_AUSTERE = SubgroupProfile(
    name="military_austere",
    description="Military/austere environment — no imaging, point-of-care biomarkers only",
    gcs_range=(3, 15),
    marshall_range=(1, 6),
    physiology=PhysiologyProfile(icp_mean=14.0, icp_std=5.0, map_mean=82.0),
    biomarker=BiomarkerProfile(
        gfap_cmax_mean=120.0, gfap_cmax_std=100.0,
        uchl1_cmax_mean=450.0, uchl1_cmax_std=300.0,
    ),
    demographics=DemographicProfile(
        age_mean=28.0, age_std=6.0, age_min=18.0, age_max=45.0,
        male_fraction=0.90,
        mechanism_weights={"fall": 0.05, "rta": 0.15, "assault": 0.05,
                           "sport": 0.0, "blast": 0.65, "other": 0.10},
        anticoagulant_rate=0.01, intoxication_rate=0.05,
    ),
    outcomes=OutcomeProfile(
        gose_weights=[0.08, 0.08, 0.10, 0.12, 0.15, 0.17, 0.15, 0.15],
        mortality_rate=0.12, icu_admission_rate=0.50, neurosurgery_rate=0.10,
    ),
    imaging_available=False,
    biomarker_available=True,
    max_observation_hours=24.0,
    observation_density=0.5,
)

ELDERLY_TBI = SubgroupProfile(
    name="elderly_tbi",
    description="Elderly TBI (age ≥65), high frailty, anticoagulation common",
    gcs_range=(3, 15),
    marshall_range=(1, 6),
    physiology=PhysiologyProfile(
        icp_mean=14.0, icp_std=5.0, map_mean=95.0, map_std=15.0,
        cbf_mean=40.0, cbf_std=10.0,
    ),
    biomarker=BiomarkerProfile(
        gfap_cmax_mean=200.0, gfap_cmax_std=150.0,
        uchl1_cmax_mean=600.0, uchl1_cmax_std=400.0,
        nfl_cmax_mean=100.0, nfl_cmax_std=80.0,
    ),
    demographics=DemographicProfile(
        age_mean=76.0, age_std=7.0, age_min=65.0, age_max=100.0,
        male_fraction=0.55,
        mechanism_weights={"fall": 0.70, "rta": 0.10, "assault": 0.03,
                           "sport": 0.02, "blast": 0.0, "other": 0.15},
        anticoagulant_rate=0.35, intoxication_rate=0.05,
    ),
    outcomes=OutcomeProfile(
        gose_weights=[0.15, 0.12, 0.15, 0.15, 0.13, 0.12, 0.10, 0.08],
        mortality_rate=0.25, icu_admission_rate=0.50, neurosurgery_rate=0.20,
    ),
    max_observation_hours=168.0,
    observation_density=1.0,
)

# Registry for lookup by name
PROFILE_REGISTRY: dict[str, SubgroupProfile] = {
    "adult_mild_tbi": ADULT_MILD_TBI,
    "adult_moderate_severe_tbi": ADULT_MODERATE_SEVERE_TBI,
    "pediatric_tbi": PEDIATRIC_TBI,
    "lmic_context": LMIC_CONTEXT,
    "military_austere": MILITARY_AUSTERE,
    "elderly_tbi": ELDERLY_TBI,
}


def get_profile(name: str) -> SubgroupProfile:
    """Retrieve a subgroup profile by name."""
    if name not in PROFILE_REGISTRY:
        available = ", ".join(PROFILE_REGISTRY.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return PROFILE_REGISTRY[name]


def list_profiles() -> list[str]:
    """List all available profile names."""
    return list(PROFILE_REGISTRY.keys())
