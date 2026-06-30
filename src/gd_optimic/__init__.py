"""
gd_optimic: Gradient descent optimization for initial conditions (IC) in ocean forecasting.

This package implements ML-4DVar optimization using a frozen pretrained ocean model (glonet v1).
Phase 1 scope: J_obs only (observation term), no background or model error terms yet.

Modules:
    data: Dataset loading, regridding, observation operators
    loss: Loss computation with observation masking
    gradient: Gradient filtering and scheduled multigrid
    optimizer: Main optimization loop with TensorBoard logging
    metrics: RMSE (global + basin-stratified) and PSD metrics
    utils: Helper functions for masks, normalizers, forward pass
    output_handler: state NetCDF + diagnostic visualization outputs
"""

__version__ = "0.1.0"

# Explicit imports for clean API
from .data import GlonetDataset, ObservationOperator
from .loss import ObservationLoss
from .gradient import GradientFilter, ScheduledPooling
from .optimizer import ICOptimizer
from .metrics import MetricsComputer
from .utils import MaskBuilder, ForwardModel
from .output_handler import OutputHandler

__all__ = [
    "GlonetDataset",
    "ObservationOperator", 
    "ObservationLoss",
    "GradientFilter",
    "ScheduledPooling",
    "ICOptimizer",
    "MetricsComputer",
    "MaskBuilder",
    "ForwardModel",
    "OutputHandler",
]
