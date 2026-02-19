"""Tests for hard physiological constraints."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constraints.monro_kellie import MonroKellieConstraint
from constraints.cerebral_perfusion import (
    CerebralPerfusionConstraint,
    ICP_TREATMENT_THRESHOLD,
    CPP_CRITICAL_LOW,
)
from constraints.physiological_bounds import PhysiologicalBounds


class TestMonroKellie:
    def test_within_budget_no_change(self):
        mk = MonroKellieConstraint(v_cranium=1400.0, v_brain_fixed=1200.0)
        # Budget = 200 mL, V_csf=100 + V_cbv=50 = 150 < 200
        state = torch.tensor([[12.0, 100.0, 50.0, 1.6, 50.0, 5.0, 50.0]])
        projected = mk(state)
        assert torch.allclose(projected, state)

    def test_over_budget_scales_down(self):
        mk = MonroKellieConstraint(v_cranium=1400.0, v_brain_fixed=1200.0)
        # Budget = 200, V_csf=150 + V_cbv=100 = 250 > 200
        state = torch.tensor([[12.0, 150.0, 100.0, 1.6, 50.0, 5.0, 50.0]])
        projected = mk(state)
        total = projected[0, 1] + projected[0, 2]
        assert total.item() <= 200.0 + 1e-5

    def test_violation_zero_when_compliant(self):
        mk = MonroKellieConstraint()
        state = torch.tensor([[12.0, 100.0, 50.0]])
        assert mk.violation(state).item() == pytest.approx(0.0, abs=1e-5)


class TestCerebralPerfusion:
    def test_cpp_identity(self):
        cp = CerebralPerfusionConstraint()
        map_val = torch.tensor(80.0)
        icp_val = torch.tensor(15.0)
        cpp = cp.compute_cpp(map_val, icp_val)
        assert cpp.item() == pytest.approx(65.0)

    def test_cpp_non_negative(self):
        cp = CerebralPerfusionConstraint()
        cpp = cp.compute_cpp(torch.tensor(20.0), torch.tensor(50.0))
        assert cpp.item() >= 0.0

    def test_icp_crisis_flag(self):
        cp = CerebralPerfusionConstraint()
        state = torch.tensor([[25.0]])  # ICP > 22
        result = cp.forward(state, torch.tensor(80.0))
        assert result["icp_crisis"].item() == 1.0

    def test_cpp_critical_flag(self):
        cp = CerebralPerfusionConstraint()
        state = torch.tensor([[30.0]])  # ICP = 30, MAP = 80 -> CPP = 50 < 60
        result = cp.forward(state, torch.tensor(80.0))
        assert result["cpp_critical"].item() == 1.0

    def test_canonical_thresholds(self):
        assert ICP_TREATMENT_THRESHOLD == 22.0
        assert CPP_CRITICAL_LOW == 60.0


class TestPhysiologicalBounds:
    def test_clamps_out_of_range(self):
        pb = PhysiologicalBounds()
        # ICP = -5 (below 0), GFAP = 60000 (above 50000)
        state = torch.tensor([[-5.0, 120.0, 50.0, 1.6, 50.0, 60000.0, 50.0]])
        projected = pb(state)
        assert projected[0, 0].item() >= 0.0
        assert projected[0, 5].item() <= 50000.0

    def test_valid_state_unchanged(self):
        pb = PhysiologicalBounds()
        state = torch.tensor([[12.0, 120.0, 50.0, 1.6, 50.0, 5.0, 50.0]])
        projected = pb(state)
        assert torch.allclose(projected, state)

    def test_violation_zero_for_valid(self):
        pb = PhysiologicalBounds()
        state = torch.tensor([[12.0, 120.0, 50.0, 1.6, 50.0, 5.0, 50.0]])
        assert pb.violation(state).item() == pytest.approx(0.0, abs=1e-5)

    def test_violation_report(self):
        pb = PhysiologicalBounds()
        state = torch.tensor([-5.0, 120.0, 50.0, 1.6, 50.0, 5.0, 50.0])
        report = pb.get_violation_report(state)
        assert len(report) == 1
        assert report[0]["variable"] == "ICP"
