"""
Bi-level meta-learning for IC optimization.

S(θ, x) learns to predict IC updates during the main optimization loop.

Main loop: x^(k+1) = x^(k) + S(θ, x^(k))
Meta-loop: θ ← θ - α_meta·∇_θ L_meta
"""

from .network_s import UNetMetaGrad2D, NetworkSConfig
from .bilevel_optimizer import BiLevelICOptimizer

__all__ = [
    "UNetMetaGrad2D",
    "NetworkSConfig", 
    "BiLevelICOptimizer",
]
