"""
Physics-Informed Physiological Trajectory Generator

Uses the mechanistic ODE systems (ICP dynamics, cerebral autoregulation)
as the generative model. Samples new trajectories by varying initial
conditions and patient parameters drawn from SubgroupProfile distributions.

Key design: the same ODE codebase serves both the forward model (prediction)
and the generative model (synthetic data), with shared constraint enforcement.
"""

import numpy as np
import torch
import pandas as pd
from typing import Optional

try:
    from ode_systems.icp_dynamics import ICPDynamics
    from ode_systems.cerebral_autoregulation import CerebralAutoregulation
    from constraints.physiological_bounds import PhysiologicalBounds, DEFAULT_BOUNDS
    from constraints.cerebral_perfusion import CerebralPerfusionConstraint
except ImportError:  # pragma: no cover - fallback for package-relative execution
    from ..ode_systems.icp_dynamics import ICPDynamics
    from ..ode_systems.cerebral_autoregulation import CerebralAutoregulation
    from ..constraints.physiological_bounds import PhysiologicalBounds, DEFAULT_BOUNDS
    from ..constraints.cerebral_perfusion import CerebralPerfusionConstraint

try:
    from synthetic.subgroup_profiles import SubgroupProfile, ADULT_MODERATE_SEVERE_TBI
except ImportError:  # pragma: no cover - fallback for package-relative execution
    from .subgroup_profiles import SubgroupProfile, ADULT_MODERATE_SEVERE_TBI


class PhysiologyGenerator:
    """
    Generate synthetic physiological time series (ICP, MAP, CPP, CBF, CVR)
    using mechanistic ODE integration with inter-patient variability.

    The generator:
      1. Samples initial conditions from the profile's distributions
      2. Integrates the ODE system forward in time
      3. Applies physiological constraint projections
      4. Adds realistic measurement noise
      5. Introduces irregular sampling / missingness
    """

    def __init__(
        self,
        profile: SubgroupProfile | None = None,
        seed: int = 42,
        dt_hours: float = 0.25,
        noise_scale: float = 1.0,
    ):
        self.profile = profile or ADULT_MODERATE_SEVERE_TBI
        self.rng = np.random.default_rng(seed)
        self.dt = dt_hours
        self.noise_scale = noise_scale

        self.bounds = PhysiologicalBounds()
        self.cpp_constraint = CerebralPerfusionConstraint()

    def _sample_initial_conditions(self) -> dict[str, float]:
        """Sample patient-specific initial conditions from profile distributions."""
        p = self.profile.physiology
        return {
            "ICP": float(np.clip(self.rng.normal(p.icp_mean, p.icp_std), 0, 80)),
            "V_csf": float(np.clip(self.rng.normal(p.v_csf_mean, p.v_csf_std), 20, 250)),
            "V_cbv": float(np.clip(self.rng.normal(p.v_cbv_mean, p.v_cbv_std), 10, 120)),
            "CVR": float(np.clip(self.rng.normal(p.cvr_mean, p.cvr_std), 0.3, 10.0)),
            "CBF": float(np.clip(self.rng.normal(p.cbf_mean, p.cbf_std), 5, 120)),
        }

    def _sample_map_trajectory(self, n_steps: int) -> np.ndarray:
        """Generate a realistic MAP trajectory with slow drift and variability."""
        p = self.profile.physiology
        base_map = self.rng.normal(p.map_mean, p.map_std)

        # Slow drift (Ornstein-Uhlenbeck process)
        map_traj = np.zeros(n_steps)
        map_traj[0] = base_map
        theta = 0.05   # mean reversion rate
        sigma = 2.0    # volatility

        for i in range(1, n_steps):
            map_traj[i] = (
                map_traj[i - 1]
                + theta * (base_map - map_traj[i - 1]) * self.dt
                + sigma * np.sqrt(self.dt) * self.rng.normal()
            )

        return np.clip(map_traj, 40, 180)

    def _integrate_ode(
        self,
        initial: dict[str, float],
        map_trajectory: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """
        Simple Euler integration of the ICP + autoregulation system.

        Returns array of shape [n_steps, 5] for [ICP, V_csf, V_cbv, CVR, CBF].
        Uses Euler method for speed in generation (not training).
        """
        icp_ode = ICPDynamics(augment_dim=0)
        autoreg_ode = CerebralAutoregulation(augment_dim=0)

        state = np.array([
            initial["ICP"], initial["V_csf"], initial["V_cbv"],
            initial["CVR"], initial["CBF"],
        ])

        trajectory = np.zeros((n_steps, 5))
        trajectory[0] = state

        for i in range(1, n_steps):
            t_tensor = torch.tensor(float(i * self.dt))
            map_val = torch.tensor(float(map_trajectory[i]))

            icp_state = torch.tensor(state[:3], dtype=torch.float32).unsqueeze(0)
            ar_state = torch.tensor(state[3:5], dtype=torch.float32).unsqueeze(0)
            icp_val = torch.tensor([state[0]], dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                d_icp = icp_ode(t_tensor, icp_state).squeeze(0).numpy()
                d_ar = autoreg_ode(
                    t_tensor, ar_state,
                    map_mmhg=map_val, icp_mmhg=icp_val,
                ).squeeze(0).numpy()

            d_state = np.concatenate([d_icp, d_ar])
            state = state + self.dt * d_state

            # Apply bounds
            state_t = torch.tensor(state, dtype=torch.float32)
            # Only clamp the 5 physiology variables
            state[0] = np.clip(state[0], 0, 100)      # ICP
            state[1] = np.clip(state[1], 0, 300)       # V_csf
            state[2] = np.clip(state[2], 0, 150)       # V_cbv
            state[3] = np.clip(state[3], 0.1, 20)      # CVR
            state[4] = np.clip(state[4], 0, 150)        # CBF

            trajectory[i] = state

        return trajectory

    def _add_measurement_noise(self, trajectory: np.ndarray) -> np.ndarray:
        """Add realistic measurement noise scaled by variable magnitude."""
        noise_stds = np.array([
            1.5 * self.noise_scale,   # ICP: ±1.5 mmHg
            5.0 * self.noise_scale,   # V_csf: ±5 mL
            2.0 * self.noise_scale,   # V_cbv: ±2 mL
            0.05 * self.noise_scale,  # CVR: ±0.05
            3.0 * self.noise_scale,   # CBF: ±3 mL/100g/min
        ])
        noise = self.rng.normal(0, noise_stds, size=trajectory.shape)
        noisy = trajectory + noise

        # Re-apply bounds after noise
        noisy[:, 0] = np.clip(noisy[:, 0], 0, 100)
        noisy[:, 1] = np.clip(noisy[:, 1], 0, 300)
        noisy[:, 2] = np.clip(noisy[:, 2], 0, 150)
        noisy[:, 3] = np.clip(noisy[:, 3], 0.1, 20)
        noisy[:, 4] = np.clip(noisy[:, 4], 0, 150)

        return noisy

    def _apply_irregular_sampling(
        self,
        trajectory: np.ndarray,
        map_trajectory: np.ndarray,
        times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Subsample to irregular observation times and introduce missingness.

        Returns (obs_times, obs_values, obs_map, obs_mask).
        """
        n_steps = len(times)
        density = self.profile.observation_density

        # Expected number of observations
        total_hours = times[-1] - times[0]
        n_obs = max(3, int(density * total_hours))
        n_obs = min(n_obs, n_steps)

        # Sample observation indices (sorted)
        obs_indices = np.sort(self.rng.choice(n_steps, size=n_obs, replace=False))

        obs_times = times[obs_indices]
        obs_values = trajectory[obs_indices]
        obs_map = map_trajectory[obs_indices]

        # Create missingness mask (some variables missing at some times)
        obs_mask = np.ones_like(obs_values)
        missing_rate = 0.1  # 10% missingness per variable
        for j in range(obs_values.shape[1]):
            missing = self.rng.random(len(obs_indices)) < missing_rate
            obs_mask[missing, j] = 0.0
            obs_values[missing, j] = 0.0  # zero out missing values

        return obs_times, obs_values, obs_map, obs_mask

    def generate_patient(self, patient_id: str | None = None) -> dict:
        """
        Generate a single synthetic patient's physiological trajectory.

        Returns dict with keys:
          patient_id, times, observations, map_values, mask,
          initial_conditions, gcs, profile_name
        """
        if patient_id is None:
            patient_id = f"SYN_{self.rng.integers(0, 99999):05d}"

        # Sample patient parameters
        initial = self._sample_initial_conditions()
        gcs = int(self.rng.integers(
            self.profile.gcs_range[0], self.profile.gcs_range[1] + 1
        ))

        # Generate time grid
        max_hours = self.profile.max_observation_hours
        n_steps = int(max_hours / self.dt) + 1
        times = np.linspace(0, max_hours, n_steps)

        # Generate MAP trajectory
        map_traj = self._sample_map_trajectory(n_steps)

        # Integrate ODE
        trajectory = self._integrate_ode(initial, map_traj, n_steps)

        # Add noise
        noisy_traj = self._add_measurement_noise(trajectory)

        # Irregular sampling
        obs_times, obs_values, obs_map, obs_mask = self._apply_irregular_sampling(
            noisy_traj, map_traj, times
        )

        # Compute CPP for each observation
        cpp_values = obs_map - obs_values[:, 0]  # CPP = MAP - ICP

        return {
            "patient_id": patient_id,
            "times": obs_times,
            "observations": obs_values,
            "map_values": obs_map,
            "cpp_values": cpp_values,
            "mask": obs_mask,
            "initial_conditions": initial,
            "gcs": gcs,
            "profile_name": self.profile.name,
            "variable_names": ["ICP", "V_csf", "V_cbv", "CVR", "CBF"],
        }

    def generate_cohort(
        self,
        n_patients: int = 100,
        id_prefix: str = "SYN",
    ) -> pd.DataFrame:
        """
        Generate a cohort of synthetic patients as a DataFrame.

        Returns long-format DataFrame with columns:
          patient_id, time_hours, ICP, V_csf, V_cbv, CVR, CBF, MAP, CPP,
          mask_ICP, mask_V_csf, mask_V_cbv, mask_CVR, mask_CBF, gcs, profile
        """
        rows = []
        for i in range(n_patients):
            pid = f"{id_prefix}_{i:04d}"
            patient = self.generate_patient(patient_id=pid)

            for t_idx in range(len(patient["times"])):
                row = {
                    "patient_id": pid,
                    "time_hours": patient["times"][t_idx],
                    "ICP": patient["observations"][t_idx, 0],
                    "V_csf": patient["observations"][t_idx, 1],
                    "V_cbv": patient["observations"][t_idx, 2],
                    "CVR": patient["observations"][t_idx, 3],
                    "CBF": patient["observations"][t_idx, 4],
                    "MAP": patient["map_values"][t_idx],
                    "CPP": patient["cpp_values"][t_idx],
                    "mask_ICP": patient["mask"][t_idx, 0],
                    "mask_V_csf": patient["mask"][t_idx, 1],
                    "mask_V_cbv": patient["mask"][t_idx, 2],
                    "mask_CVR": patient["mask"][t_idx, 3],
                    "mask_CBF": patient["mask"][t_idx, 4],
                    "gcs": patient["gcs"],
                    "profile": patient["profile_name"],
                }
                rows.append(row)

        return pd.DataFrame(rows)
