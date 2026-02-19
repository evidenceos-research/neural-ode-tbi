"""Tests for ODE system modules."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ode_systems.icp_dynamics import ICPDynamics
from ode_systems.cerebral_autoregulation import CerebralAutoregulation
from ode_systems.biomarker_kinetics import BiomarkerKinetics, GFAP_CT_THRESHOLD, UCHL1_CT_THRESHOLD
from ode_systems.multi_compartment import MultiCompartmentTBI


class TestICPDynamics:
    def test_forward_shape(self):
        model = ICPDynamics()
        state = torch.tensor([[12.0, 120.0, 50.0]])
        t = torch.tensor(0.0)
        d_state = model(t, state)
        assert d_state.shape == state.shape

    def test_parameters_positive(self):
        model = ICPDynamics()
        assert model.compliance.item() > 0
        assert model.csf_production_rate.item() > 0
        assert model.absorption_coeff.item() > 0
        assert model.outflow_resistance.item() > 0

    def test_augmented_forward(self):
        model = ICPDynamics(augment_dim=2)
        state = torch.tensor([[12.0, 120.0, 50.0, 0.0, 0.0]])
        t = torch.tensor(0.0)
        d_state = model(t, state)
        assert d_state.shape == state.shape

    def test_param_summary(self):
        model = ICPDynamics()
        summary = model.get_param_summary()
        assert "compliance" in summary
        assert "csf_production_rate" in summary


class TestCerebralAutoregulation:
    def test_forward_shape(self):
        model = CerebralAutoregulation()
        state = torch.tensor([[1.6, 50.0]])
        t = torch.tensor(0.0)
        map_val = torch.tensor(80.0)
        icp_val = torch.tensor(12.0)
        d_state = model(t, state, map_mmhg=map_val, icp_mmhg=icp_val)
        assert d_state.shape == state.shape

    def test_autoreg_integrity_bounded(self):
        model = CerebralAutoregulation()
        integrity = model.autoreg_integrity.item()
        assert 0.0 <= integrity <= 1.0


class TestBiomarkerKinetics:
    def test_forward_shape(self):
        model = BiomarkerKinetics()
        state = torch.tensor([[5.0, 50.0]])
        t = torch.tensor(0.0)
        d_state = model(t, state)
        assert d_state.shape == state.shape

    def test_injury_signal_decays(self):
        model = BiomarkerKinetics()
        s0 = model.injury_signal(torch.tensor(0.0))
        s10 = model.injury_signal(torch.tensor(10.0))
        assert s10.item() < s0.item()

    def test_threshold_evaluation(self):
        model = BiomarkerKinetics()
        result = model.evaluate_thresholds(gfap=50.0, uchl1=400.0)
        assert result["ct_indicated"] is True
        assert result["gfap_positive"] is True
        assert result["uchl1_positive"] is True

    def test_threshold_negative(self):
        model = BiomarkerKinetics()
        result = model.evaluate_thresholds(gfap=10.0, uchl1=100.0)
        assert result["ct_indicated"] is False
        assert result["biomarker_negative"] is True

    def test_canonical_thresholds(self):
        assert GFAP_CT_THRESHOLD == 30.0
        assert UCHL1_CT_THRESHOLD == 360.0

    def test_default_half_life_ranges(self):
        model = BiomarkerKinetics()
        summary = model.get_param_summary()
        assert 24.0 <= summary["half_life_gfap_hours"] <= 36.0
        assert 7.0 <= summary["half_life_uchl1_hours"] <= 9.0


class TestMultiCompartmentTBI:
    def test_forward_shape(self):
        model = MultiCompartmentTBI()
        state = MultiCompartmentTBI.default_initial_state(batch_size=2)
        t = torch.tensor(0.0)
        d_state = model(t, state)
        assert d_state.shape == state.shape

    def test_state_dim(self):
        assert MultiCompartmentTBI.STATE_DIM == 7
        assert len(MultiCompartmentTBI.STATE_NAMES) == 7

    def test_default_initial_state(self):
        state = MultiCompartmentTBI.default_initial_state(batch_size=3)
        assert state.shape == (3, 7)
        assert state[:, 0].mean().item() == pytest.approx(12.0)  # ICP
        assert state[:, 4].mean().item() == pytest.approx(50.0)  # CBF

    def test_augmented_forward(self):
        model = MultiCompartmentTBI(augment_dim=4)
        state = MultiCompartmentTBI.default_initial_state(batch_size=1, augment_dim=4)
        t = torch.tensor(0.0)
        d_state = model(t, state)
        assert d_state.shape == state.shape

    def test_param_summary(self):
        model = MultiCompartmentTBI()
        summary = model.get_param_summary()
        assert "icp_to_biomarker_gain" in summary
        assert "icp.compliance" in summary
        assert "autoreg.cbf_target" in summary
        assert "biomarker.k_release_gfap" in summary
