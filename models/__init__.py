"""
Neural ODE Model Wrappers

- HybridNeuralODE: Physics-informed hybrid model (primary)
- LatentODE: Latent ODE baseline comparator
"""

from .hybrid_ode import HybridNeuralODE
from .latent_ode import LatentODE

__all__ = ["HybridNeuralODE", "LatentODE"]
