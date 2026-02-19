"""Tests for Neural ODE model wrappers."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.hybrid_ode import HybridNeuralODE
from models.latent_ode import LatentODE


class TestHybridNeuralODE:
    def test_forward_produces_predictions(self):
        model = HybridNeuralODE(obs_dim=7, augment_dim=2)
        first_obs = torch.randn(1, 7)
        times = torch.linspace(0, 10, 5)
        result = model(first_obs, times)
        assert "predictions" in result
        assert result["predictions"].shape == (5, 1, 7)

    def test_states_shape(self):
        model = HybridNeuralODE(obs_dim=7, augment_dim=2)
        first_obs = torch.randn(1, 7)
        times = torch.linspace(0, 10, 3)
        result = model(first_obs, times)
        assert result["states"].shape[0] == 3
        assert result["states"].shape[1] == 1

    def test_constraint_violations_tracked(self):
        model = HybridNeuralODE(obs_dim=7, augment_dim=0)
        first_obs = torch.randn(1, 7)
        times = torch.linspace(0, 5, 3)
        result = model(first_obs, times)
        assert len(result["constraint_violations"]) == 3

    def test_compute_loss(self):
        model = HybridNeuralODE(obs_dim=7, augment_dim=0)
        preds = torch.randn(5, 1, 7)
        targets = torch.randn(5, 1, 7)
        losses = model.compute_loss(preds, targets)
        assert "total_loss" in losses
        assert "reconstruction_loss" in losses
        assert losses["total_loss"].item() > 0

    def test_compute_loss_with_mask(self):
        model = HybridNeuralODE(obs_dim=7, augment_dim=0)
        preds = torch.randn(5, 1, 7)
        targets = torch.randn(5, 1, 7)
        mask = torch.ones(5, 1, 7)
        mask[2:, :, 3:] = 0  # mask out some observations
        losses = model.compute_loss(preds, targets, mask=mask)
        assert losses["total_loss"].item() > 0


class TestLatentODE:
    def test_forward_produces_predictions(self):
        model = LatentODE(obs_dim=7, latent_dim=8)
        obs = torch.randn(2, 5, 7)  # batch=2, T=5
        obs_times = torch.linspace(0, 10, 5)
        pred_times = torch.linspace(0, 15, 8)
        result = model(obs, obs_times, pred_times)
        assert result["predictions"].shape == (8, 2, 7)

    def test_latent_states_shape(self):
        model = LatentODE(obs_dim=7, latent_dim=16)
        obs = torch.randn(1, 3, 7)
        obs_times = torch.linspace(0, 5, 3)
        pred_times = torch.linspace(0, 10, 6)
        result = model(obs, obs_times, pred_times)
        assert result["latent_states"].shape == (6, 1, 16)

    def test_compute_loss(self):
        model = LatentODE(obs_dim=7)
        preds = torch.randn(5, 1, 7)
        targets = torch.randn(5, 1, 7)
        losses = model.compute_loss(preds, targets)
        assert "total_loss" in losses
        assert losses["total_loss"].item() > 0
