"""
IC optimizer selection (feature flag: optimization.use_ic_optimizer).

Provides:
    - SUPPORTED_IC_OPTIMIZERS: names accepted in optimization.ic_optimizer.name
    - build_ic_optimizer: torch.optim optimizer acting on the IC tensor

When the flag is off, ICOptimizer keeps its constant-step update
x <- x - learning_rate * masked_gradient. When it is on, the same masked
(filtered, ocean-masked) gradient is fed to a torch.optim optimizer instead,
so Adam & co. are compared on exactly the same gradient signal.
"""

import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

# name -> (torch.optim class, keys of the ic_optimizer config forwarded as kwargs)
_OPTIMIZER_REGISTRY = {
    "sgd": (torch.optim.SGD, ("momentum", "dampening", "nesterov", "weight_decay")),
    "adam": (torch.optim.Adam, ("betas", "eps", "weight_decay", "amsgrad")),
    "adamw": (torch.optim.AdamW, ("betas", "eps", "weight_decay", "amsgrad")),
    "rmsprop": (torch.optim.RMSprop, ("alpha", "eps", "momentum", "centered", "weight_decay")),
    "adagrad": (torch.optim.Adagrad, ("lr_decay", "eps", "weight_decay")),
    "nadam": (torch.optim.NAdam, ("betas", "eps", "weight_decay", "momentum_decay")),
}

SUPPORTED_IC_OPTIMIZERS = tuple(_OPTIMIZER_REGISTRY)


def resolve_ic_optimizer_lr(ic_optimizer_config: Dict, default_lr: float) -> float:
    """Return ic_optimizer.lr if set, else the global optimization.learning_rate.

    Adaptive methods (Adam, RMSprop, ...) normalize the gradient, so their lr is a
    per-element step size in IC units and usually needs a very different value
    from the constant-step learning_rate -- hence the separate override.
    """
    lr = ic_optimizer_config.get("lr")
    return float(default_lr if lr is None else lr)


def build_ic_optimizer(
    x0: torch.Tensor,
    ic_optimizer_config: Dict,
    default_lr: float,
) -> torch.optim.Optimizer:
    """
    Build a torch.optim optimizer over the IC tensor.

    Args:
        x0: Leaf IC tensor [B, T, C, H, W] with requires_grad=True (updated in place)
        ic_optimizer_config: optimization.ic_optimizer config dict with keys:
            - name: one of SUPPORTED_IC_OPTIMIZERS (default: 'adam')
            - lr: step size; null falls back to default_lr
            - optimizer-specific kwargs (betas, eps, momentum, weight_decay, ...);
              null values are ignored so torch defaults apply
        default_lr: optimization.learning_rate

    Returns:
        Configured torch.optim.Optimizer
    """
    name = str(ic_optimizer_config.get("name", "adam")).lower()
    if name not in _OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown ic_optimizer.name '{name}'. Supported: {', '.join(SUPPORTED_IC_OPTIMIZERS)}"
        )

    optimizer_cls, allowed_keys = _OPTIMIZER_REGISTRY[name]
    kwargs = {}
    for key in allowed_keys:
        value = ic_optimizer_config.get(key)
        if value is None:
            continue
        if key == "betas":
            value = tuple(float(b) for b in value)
        kwargs[key] = value

    lr = resolve_ic_optimizer_lr(ic_optimizer_config, default_lr)
    optimizer = optimizer_cls([x0], lr=lr, **kwargs)
    logger.info(f"IC optimizer: {optimizer_cls.__name__}(lr={lr}, {kwargs})")
    return optimizer
