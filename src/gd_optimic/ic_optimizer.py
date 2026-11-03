"""
IC optimizer selection (feature flag: optimization.use_ic_optimizer).

Provides:
    - SUPPORTED_IC_OPTIMIZERS: names accepted in optimization.ic_optimizer.name
    - build_ic_optimizer: torch.optim optimizer acting on the IC tensor
    - ReferenceStep: the outer step f(ψ^k) (GD or torch.optim) computed without moving
      the IC, followed by the meta-learner in Δ inner-loss mode

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


class ReferenceStep:
    """
    The outer update rule f of  ψ^{k+1} = ψ^k − f(ψ^k), as a stand-alone callable.

    In plain words: this computes "the step the baseline optimizer would take from ψ^k
    with gradient g^k" WITHOUT moving ψ^k. The meta-learner (Δ inner-loss mode) uses it
    as the reference S has to follow, so S is compared fairly with whatever baseline
    is configured:
        - ic_optimizer_config is None  -> plain GD:   f = learning_rate · g
        - otherwise (adagrad, adam...) -> f = the torch.optim step on g, with its own
                                          state (Adagrad's Σg², Adam's moments...)

    Call it exactly once per outer gradient: the optimizer state then evolves exactly
    as in the baseline run (for Adagrad: G += g² once per outer iteration).

    How it works for torch optimizers: the optimizer owns a private "shadow" copy of the
    IC. Each call copies ψ^k into it, takes one optimizer step with g as its gradient,
    and reads the step back as f = ψ^k − shadow. The real IC is never touched. (Copying
    ψ^k in first keeps weight_decay exact, since it depends on the parameter value.)
    """

    def __init__(
        self,
        x_like: torch.Tensor,
        ic_optimizer_config: Optional[Dict],
        learning_rate: float,
    ):
        """
        Args:
            x_like: IC tensor [B, T, C, H, W] (only its shape/dtype/device are used)
            ic_optimizer_config: optimization.ic_optimizer dict, or None for plain GD
            learning_rate: optimization.learning_rate (GD step, and default lr of the optimizer)
        """
        self.learning_rate = float(learning_rate)
        self.optimizer = None
        self._shadow = None
        if ic_optimizer_config is not None:
            # Private leaf tensor the optimizer updates in place (its state is keyed on it)
            self._shadow = x_like.detach().clone().requires_grad_(True)
            self.optimizer = build_ic_optimizer(self._shadow, ic_optimizer_config, self.learning_rate)

    def describe(self) -> str:
        """Short name for logs, e.g. 'GD(lr=0.01)' or 'Adagrad(lr=0.1)'."""
        if self.optimizer is None:
            return f"GD(lr={self.learning_rate})"
        lr = self.optimizer.param_groups[0]["lr"]
        return f"{type(self.optimizer).__name__}(lr={lr})"

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        gradient: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return f(ψ^k) for the current IC x = ψ^k and its (masked) gradient g^k.

        Args:
            x: current IC ψ^k [B, T, C, H, W]
            gradient: masked outer gradient g^k, same shape
            mask: optional ocean mask broadcastable to x; the step is re-masked so land
                  points never move (same as the baseline IC-optimizer branch)

        Returns:
            f: step in IC units, so that the reference update is ψ^{k+1} = ψ^k − f
        """
        if self.optimizer is None:
            step = self.learning_rate * gradient
        else:
            x = x.detach()
            self._shadow.copy_(x)
            self._shadow.grad = gradient.detach().clone()
            self.optimizer.step()
            step = x - self._shadow
            self._shadow.grad = None
        if mask is not None:
            step = step * mask
        return step.detach()
