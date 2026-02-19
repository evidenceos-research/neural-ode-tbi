"""Regression tests for synthetic biomarker generator consistency."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthetic.biomarker_generator import (  # noqa: E402
    BiomarkerGenerator,
    GFAP_HALF_LIFE_RANGE_HOURS,
    UCHL1_HALF_LIFE_RANGE_HOURS,
)


class TestSyntheticBiomarkerGenerator:
    def test_sampled_half_lives_within_canonical_ranges(self):
        gen = BiomarkerGenerator(seed=123)
        for _ in range(200):
            params = gen._sample_pk_params()
            assert GFAP_HALF_LIFE_RANGE_HOURS[0] <= params["GFAP"]["half_life_hours"] <= GFAP_HALF_LIFE_RANGE_HOURS[1]
            assert UCHL1_HALF_LIFE_RANGE_HOURS[0] <= params["UCH_L1"]["half_life_hours"] <= UCHL1_HALF_LIFE_RANGE_HOURS[1]

    def test_optional_analytes_included_when_enabled(self):
        gen = BiomarkerGenerator(seed=99, include_s100b=True, include_nfl=True)
        patient = gen.generate_patient(times_hours=np.array([0.0, 6.0, 24.0]))
        assert {"GFAP", "UCH_L1", "S100B", "NfL"}.issubset(set(patient["analytes"].keys()))

    def test_ct_indicated_logic_is_boolean_or(self):
        gen = BiomarkerGenerator(seed=7)
        patient = gen.generate_patient(times_hours=np.array([0.0, 4.0, 8.0, 12.0]))
        crossings = patient["threshold_crossings"]
        assert crossings["ct_indicated"] == (
            crossings["gfap_ct_positive"] or crossings["uchl1_ct_positive"]
        )
