"""
Bi-level optimization loop: IC optimization + meta-learner adaptation.

Inner loop: Update IC using S(θ, x)
Outer loop: Update S's parameters θ using meta-loss
"""

from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
import logging
import numpy as np

from .network_s import UNetMetaGrad2D, NetworkSConfig



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def hvp_efficient(
    loss: torch.Tensor,
    params: List[torch.nn.Parameter],
    v: torch.Tensor,
    damping: float = 0.0,
) -> Tuple[List[torch.Tensor], float]:
    """
    Compute Hessian-vector product (HVP) efficiently using autodiff.

    Computes: (∇²_x loss) @ v (Hessian times vector product)

    This is memory-efficient for second-order derivatives because:
    - We compute (∂²loss/∂x²) @ v directly without materializing the full Hessian
    - Avoids storing the full computational graph
    - Scales linearly with model size, not quadratically

    Args:
        loss: Scalar loss value (must have requires_grad=True)
        params: List of parameters to compute Hessian w.r.t.
        v: Vector to multiply with Hessian (same shape as loss gradient)
        damping: Damping factor λ for numerical stability: (H + λI) @ v

    Returns:
        hvp_result: List of Hessian-vector products (one per param)
        hvp_norm: L2 norm of HVP for diagnostics
    """
    # Compute gradient (first-order)
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

    # Compute vector-Jacobian product: (∂grads/∂x) @ v
    # This gives us the Hessian-vector product without materializing H
    hvp = torch.autograd.grad(
        grads,
        params,
        grad_outputs=v,
        retain_graph=False,
        allow_unused=True,
    )

    # Apply damping: (H + λI) @ v ≈ hvp + λ*v
    if damping > 0:
        hvp = tuple(
            h + damping * v_i if h is not None else damping * v_i
            for h, v_i in zip(hvp, v)
        )

    # Compute norm for diagnostics
    hvp_norm = torch.norm(torch.cat([h.flatten() for h in hvp if h is not None]))

    return hvp, float(hvp_norm.cpu().item())





class BiLevelICOptimizer:
    """
    Bi-level optimizer for IC optimization with learned update rules.

    Main loop:
        x^(k+1) = x^(k) + S(θ, x^(k))

    Meta loop:
        θ ← θ - α_meta·∇_θ L_meta

    where L_meta = w₁·L_perf + w₂·L_smooth + λ||θ||²

    Memory Optimization (OOM Prevention):
    ====================================
    Second-order gradients (dJ/dθ/dx) can cause massive OOM in large models.

    Solutions implemented:
    1. **Disabled by default**: use_grad_smooth=False (gradient smoothness term disabled)
    2. **No create_graph=True**: Avoids building full second-order computational graphs
    3. **First-order approximation**: ALIGN phase uses only first-order gradients
    4. **Activation checkpointing**: Forward pass can use recompute-on-backward
    5. **Aggressive cache cleanup**: torch.cuda.empty_cache() after each meta-step

    If you enable gradient smoothness (use_grad_smooth=True), the ALIGN phase will:
    - Compute grad_current WITHOUT create_graph=True
    - Use gradient norm as smoothness proxy (no second-order autodiff)
    - This trades off theoretical purity for practical memory efficiency

    For full second-order optimization on large models, consider:
    - Reducing batch size or sequence length
    - Using parameter-efficient adapters (LoRA, etc.)
    - Implicit differentiation methods (Neumann series approximation)
    """

    def __init__(
        self,
        network_s: UNetMetaGrad2D,
        ocean_mask: torch.Tensor,
        forward_model,
        loss_fn,
        num_forecast_steps: int,
        device: str = "cuda",
        config: Optional[DictConfig] = None,
        writer=None,
    ):
        """
        Initialize bi-level optimizer for meta-learned IC optimization.

        Two-phase curriculum learning:
        1. ALIGN phase (first grad_align_steps iterations): Learn gradient-aligned patterns
           - Network S learns to predict updates that align with observed gradients ∇_x
           - Objective: l_align + l_reg
           - Teaches the SHAPE of good updates (physical consistency)
           - l_align = ||S(θ, x) - ∇_x||_2 measures alignment with observed gradient

        2. PERF phase (remaining iterations): Maximize loss reduction
           - Network S refines learned updates to actually reduce forecast loss
           - Objective: w_perf * L_perf + l_reg (± weak l_align regularizer)
           - Focuses on PERFORMANCE (gradient descent effectiveness)
           - L_perf = loss(forecast with updated IC) measures forecast improvement

        Key hyperparameters:
        - w_align: Weight on gradient alignment (controls ALIGN phase learning)
        - w_perf: Weight on loss minimization (controls PERF phase learning)
        - lambda_reg: Weight on L2 regularization ||θ||² (prevents parameter explosion)

        Args:
            network_s: S(θ, x) network for IC update prediction
                - Input: Current IC state [B, T, C, H, W]
                - Output: Predicted IC update [B, T, C, H, W]
            ocean_mask: [H, W] binary ocean mask (1=ocean, 0=land)
            forward_model: Forward model for forecast computation
            loss_fn: Loss function: J = loss(forecast, target)
            num_forecast_steps: Number of time steps to forecast
            device: torch device (cuda/cpu)
            config: Hydra config with hyperparameters:
                - meta_lr: Learning rate for S's parameters (default: 1e-3, typically 1e-4)
                - num_meta_steps: IC update iterations per outer loop (default: 10)
                - grad_align_steps: Duration of ALIGN phase in meta-steps (default: 6)
                - w_align: Alignment loss weight (default: 1.0)
                - w_perf: Performance loss weight (default: 1.0)
                - lambda_reg: L2 regularization weight (default: 1e-4)
                - align_lr_multiplier: ALIGN-phase meta learning-rate multiplier (default: 1.0)
                - trans_lr_multiplier: TRANS-phase meta learning-rate multiplier (default: 1.0)
            writer: TensorBoard SummaryWriter for logging meta-step metrics
        """
        self.network_s = network_s.to(device)
        self.ocean_mask = ocean_mask.to(device).unsqueeze(0)  # [1, H, W]
        self.forward_model = forward_model
        self.loss_fn = loss_fn
        self.num_forecast_steps = num_forecast_steps
        self.device = device
        self.writer = writer
        self.channel_mean, self.channel_std = self._get_channel_statistics()

        # Configuration
        if config is None:
            config = DictConfig({})

        self.config = config
        self.enabled = config.get("enabled", True)

        # Curriculum: three-phase learning
        self.grad_perf_steps = config.get("grad_perf_steps", 6)  # Number of initial meta-steps to focus on alignment
        self.grad_trans_steps = config.get("grad_trans_steps", 2)  # Number of transition steps (between ALIGN and PERF)

        # Meta-loss weights
        self.w_align = config.get("w_align", 1.0)   # Alignment loss weight (in both phases)
        self.w_perf = config.get("w_perf", 1.0)     # Performance loss weight (PERF phase)
        self.w_trans = config.get("w_trans", 0.5)   # Transition phase alignment weight (weaker)
        self.lambda_reg = config.get("lambda_reg", 1e-4)  # Regularization (was 1e-5, but 1e-4 better)

        # Meta-optimizer
        self.meta_lr = config.get("meta_lr", 1e-3)
        self.num_meta_steps = config.get("num_meta_steps", 10)  # ← NEW: multiple meta updates per IC update
        self.align_lr_multiplier = float(config.get("align_lr_multiplier", 1.0))
        self.trans_lr_multiplier = float(config.get("trans_lr_multiplier", 1.0))
        self.meta_optimizer = optim.Adam(
            self.network_s.parameters(),
            lr=self.meta_lr,
            weight_decay=1e-2,
        )
        self._meta_base_lrs = [group["lr"] for group in self.meta_optimizer.param_groups]

        # Gradient clipping norm (configurable)
        self.grad_clip_norm = float(config.get("grad_clip_norm", 1.0))

        # Gradient accumulation (since IC update happens first, we defer meta-update)
        self.pending_meta_loss = None

        # Target sequence (set at each step call)
        self.target_sequence = None

        # Logging
        self.update_history = []
        self.loss_history = []
        self.meta_loss_history = []

    def step(
        self,
        first_update: torch.Tensor,
        x_current: torch.Tensor,
        loss_prev: torch.Tensor,
        gradient_prev: torch.Tensor,
        iteration: int = 0,
    ) -> Dict[str, float]:
        """
        Execute one meta-learning step (or multiple if num_meta_steps > 1).

        This performs bi-level optimization:
        - Inner loop: Update IC using learned S(θ, x)
        - Outer loop (meta): Update S's parameters θ to minimize meta-loss

        Three-phase curriculum:

        **ALIGN Phase (m < grad_align_steps):**
        - Goal: Train network S to predict gradient-aligned updates
        - Loss: w_align * l_align + l_reg
        - l_align = ||S(θ, x) - ∇_x J||_2 (alignment with observed gradient)
        - Teaches S to recognize patterns that match observed gradients
        - Should see: l_align DECREASING (network learning gradient patterns)

        **TRANS Phase (grad_align_steps <= m < grad_align_steps + grad_trans_steps):**
        - Goal: Transition from gradient alignment to performance optimization
        - Loss: w_trans * l_align + w_perf * l_perf + l_reg (balanced)
        - Gradually shifts focus from alignment to actual loss reduction
        - Stabilizes model parameters before aggressive performance optimization
        - Should see: Both l_align and l_perf starting to decrease together

        **PERF Phase (m >= grad_align_steps + grad_trans_steps):**
        - Goal: Refine S to maximize actual loss reduction
        - Loss: w_perf * L_perf + l_reg
        - L_perf = loss_current = forecast error at updated IC
        - Should see: L_perf DECREASING, improvement POSITIVE (IC updates work!)

        Key metrics logged to TensorBoard:
        - L_align: Gradient alignment loss (should decrease in ALIGN phase)
        - L_perf: Forecast loss at updated IC (should decrease in PERF phase)
        - improvement: loss_prev_step - loss_current (should be positive in PERF)
        - L_reg: Regularization term (should stay stable)
        - grad_norm: Gradient norm (for diagnostics)

        Gradient Flow Notes (Issue #2 fix):
        - gradient_prev is explicitly detached (line 280)
        - x_det is detached at each meta-step (no gradient propagation through trajectory)
        - ic_update naturally depends on theta via network_s (no clone() breaking gradients)
        - Only first-order gradients computed (no create_graph=True)
        - Each meta-step is independent: ∇_θ l_meta is computed fresh

        Args:
            x_current: [B, T, C, H, W] current IC state
            loss_prev: scalar, forecast loss at x_current
            gradient_prev: [B, T, C, H, W] ∇_x loss, computed via backprop through forward_model
                - Explicitly detached to avoid OOM during meta-optimization
                - Represents the direction that reduces loss via gradient descent
            first_update: [B, T, C, H, W] IC update for first meta-step
            iteration: Current outer loop iteration (for TensorBoard logging)

        Returns:
            diagnostics: Dict with keys:
                - L_align: Gradient alignment loss
                - L_perf: Forecast loss at updated IC
                - L_meta: Combined meta-loss (used for backward pass)
                - L_reg: L2 regularization penalty
                - improvement: Loss reduction from previous step
        """

        loss_prev = loss_prev.to(self.device) if torch.is_tensor(loss_prev) else torch.tensor(float(loss_prev), device=self.device)
        gradient_prev = gradient_prev.to(self.device).detach()
        gradient_mean, gradient_std = self._get_gradient_statistics(gradient_prev)

        # Ensure network_s is in training mode
        self.network_s.train()

        # Diagnostic: Check network_s state
        logger.debug(f"network_s training mode: {self.network_s.training}")
        param_count = sum(p.numel() for p in self.network_s.parameters())
        requires_grad_count = sum(p.numel() for p in self.network_s.parameters() if p.requires_grad)
        logger.debug(f"network_s parameters: {param_count} total, {requires_grad_count} with requires_grad=True")

        last_l_meta = 0.0
        last_l_align = 0.0
        last_l_perf = 0.0
        last_l_reg = 0.0

        # Track loss from previous meta-step for improvement calculation
        loss_prev_step = loss_prev

        x_det = torch.zeros_like(x_current)
        for m in range(self.num_meta_steps):
            # Determine phase: ALIGN -> TRANS -> PERF.
            # grad_perf_steps is retained as the historical config key; it
            # represents the initial ALIGN-phase duration.
            if m < self.grad_perf_steps:
                phase_name = "ALIGN"
            elif m < self.grad_perf_steps + self.grad_trans_steps:
                phase_name = "TRANS"
            else:
                phase_name = "PERF"

            # Adam is mostly invariant to multiplying the loss gradient.
            # Use an explicit phase-dependent learning rate when a larger
            # ALIGN update is desired.
            lr_multiplier = {
                "ALIGN": self.align_lr_multiplier,
                "TRANS": self.trans_lr_multiplier,
                "PERF": 1.0,
            }[phase_name]
            for group, base_lr in zip(self.meta_optimizer.param_groups, self._meta_base_lrs):
                group["lr"] = base_lr * lr_multiplier

            self.meta_optimizer.zero_grad()

            # ============================================================================
            # STEP 1: Update IC using learned network S
            # ============================================================================
            # Key principle: Each meta-step is independent gradient computation
            # - No create_graph=True (no second-order autodiff)
            # - No higher-order gradients through iterations
            # - Pure first-order gradient descent on meta-loss

            if m == 0:
                # First step: use the provided first_update
                x_prev = (x_current + first_update)
                x_det = x_prev.detach()  # Detach from previous iteration to avoid gradient accumulation
            # else:
            #     # Subsequent steps: predict new update and apply it
            #     # CRITICAL: Detach x_det so we don't propagate gradients through the entire trajectory
            #     # Each step is independent: only gradient of l_meta on theta is needed
            #     # x_det = x_det.detach().clone()  # Detach from previous iteration
            #     ic_update = self.predict_update(x_det)  # Network output naturally requires_grad via network_s parameters
            #     # ic_update depends on theta (network_s.parameters()) and x_det
            #     # This is the natural gradient flow we want

            ic_update = self.predict_update(
                x_det,
                gradient_mean=gradient_mean,
                gradient_std=gradient_std,
            )
            # Apply update: new IC = current IC - update (gradient descent on loss)
            x_new = x_det - ic_update  # Broadcasting works for [B,T,C,H,W] - [B,T,C,H,W]

            # ============================================================================
            # STEP 2: Forward pass and compute losses
            # ============================================================================
            _, y_hat_steps_new = self.forward_model.forward(x_new, self.num_forecast_steps)
            loss_current, _ = self.loss_fn(y_hat_steps_new, self.target_sequence, return_details=True)

            with torch.no_grad():
                # Performance metric: improvement from PREVIOUS meta-step
                # Positive = loss decreased (good), Negative = loss increased (bad)
                improvement = loss_prev_step - loss_current

            # ============================================================================
            # STEP 3: Compute channel-standardized alignment loss
            # ============================================================================
            # Standardize the target gradient independently for each channel.
            # The statistics are detached: no gradient is propagated through
            # the target normalization.
            gradient_standardized = (
                gradient_prev - self._broadcast_channels(gradient_mean, gradient_prev)
            ) / self._broadcast_channels(gradient_std, gradient_prev)
            predicted_gradient_standardized = (
                ic_update - self._broadcast_channels(gradient_mean, ic_update)
            ) / self._broadcast_channels(gradient_std, ic_update)
            l_align = (gradient_standardized - predicted_gradient_standardized).pow(2).mean()

            l_perf = loss_current.mean() if loss_current.dim() > 0 else loss_current
            l_reg_raw = self.lambda_reg * sum((p ** 2).sum() for p in self.network_s.parameters())
            l_reg_scalar = l_reg_raw / sum(p.numel() for p in self.network_s.parameters())

            # Diagnostic: Check gradient flow on first meta-step
            if m == 0:
                logger.debug(f"[iter: 0]: ic_update requires_grad={ic_update.requires_grad}, "
                            f"grad_fn={ic_update.grad_fn}, "
                            f"l_align requires_grad={l_align.requires_grad}, "
                            f"grad_fn={l_align.grad_fn}, "
                            f"loss_current={loss_current.item():.6f}")

            # ============================================================================
            # STEP 4: Combine meta-loss components by phase and compute gradients
            # ============================================================================

            params = list(self.network_s.parameters())

            # Safe gradient clearing (not in-place on graph)
            self.meta_optimizer.zero_grad()

            l_align_weighted = self.w_align * l_align
            l_perf_weighted = self.w_perf * l_perf
            l_reg_weighted = 1.0 * l_reg_scalar
            l_combined = l_align_weighted + l_perf_weighted + l_reg_weighted

            if phase_name == "ALIGN":
                # ALIGN: Focus on gradient alignment
                # Goal: Learn the SHAPE of good updates (physical consistency)
                l_meta = l_align_weighted

                logger.debug(f"ALIGN: l_meta={l_meta.item():.6f}")

            elif phase_name == "TRANS":
                # TRANS: Balance alignment and performance
                # Goal: Smoothly transition from alignment focus to performance focus
                l_meta = l_align_weighted + l_reg_weighted

                logger.debug(f"TRANS: l_meta={l_meta.item():.6f}")

            else:  # phase_name == "PERF"
                # PERF: Focus on performance (actual loss reduction)
                # Goal: Maximize loss reduction effectiveness
                l_meta = l_perf_weighted


                logger.debug(f"PERF: l_meta={l_meta.item():.6f}")

            # Backward pass: Compute gradients of l_meta w.r.t. network_s parameters
            # No retain_graph=True needed: each meta-step is independent
            # x_det is detached, so no trajectory gradients
            l_meta.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip_norm)

            # ============================================================================
            # STEP 5: Optimizer step and logging
            # ============================================================================
            # Check gradient magnitude before update
            grad_norm = sum(p.grad.norm().item() ** 2 for p in params if p.grad is not None) ** 0.5
            logger.debug(f"Meta-step {m+1}/{self.num_meta_steps} ({phase_name}): "
                        f"grad_norm(clipped)={grad_norm:.6f}, "
                        f"lr_multiplier={lr_multiplier:.3f}, ")

            # Adam optimizer step
            self.meta_optimizer.step()

            # Update state for next iteration
            loss_prev_step = loss_current.detach()
            x_det = x_new.detach()

            # Cache loss values for logging
            last_l_combined = float(l_combined.detach().cpu().item())
            last_l_align = float(l_align_weighted.detach().cpu().item())
            last_l_perf = float(l_perf_weighted.detach().cpu().item())
            last_l_reg = float(l_reg_weighted.detach().cpu().item())

            # TensorBoard logging
            if self.writer is not None:
                global_step = iteration * self.num_meta_steps + m + 1
                self.writer.add_scalar(f"meta_steps/l_combined", last_l_combined, global_step)
                self.writer.add_scalar(f"meta_steps/L_align", last_l_align, global_step)
                self.writer.add_scalar(f"meta_steps/L_perf", last_l_perf, global_step)
                self.writer.add_scalar(f"meta_steps/L_reg", last_l_reg, global_step)
                self.writer.add_scalar(f"meta_steps/grad_norm", grad_norm, global_step)
                self.writer.add_scalar(f"meta_steps/phase", {"ALIGN": 0, "TRANS": 1, "PERF": 2}[phase_name], global_step)

            if m % 10 == 0:
                logger.info(
                    f"Meta-step {m+1}/{self.num_meta_steps} ({phase_name}): "
                    f"L_combined={last_l_combined:.6f}, L_align={last_l_align:.6f}, "
                    f"L_perf={last_l_perf:.6f}, L_reg={last_l_reg:.6e}, ")

            # Memory cleanup
            # del loss_current, l_reg_scalar, l_reg_standardized, l_reg_weighted
            del loss_current, l_reg_scalar, l_perf
            del improvement, y_hat_steps_new, ic_update
            del x_new, l_align, l_align_weighted, l_combined, l_meta

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        diagnostics = {
            "L_combined": last_l_combined,
            "L_align": last_l_align,
            "L_perf": last_l_perf,
            "L_reg": last_l_reg,
            "num_meta_steps": self.num_meta_steps,
        }

        return diagnostics

    def _get_channel_statistics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the forward model's per-channel normalization statistics."""
        normalizer = getattr(self.forward_model, "normalizer", None)
        if normalizer is None or not hasattr(normalizer, "mean") or not hasattr(normalizer, "std"):
            raise ValueError("forward_model must expose per-channel normalizer.mean and normalizer.std")

        mean = torch.as_tensor(normalizer.mean, dtype=torch.float32, device=self.device).flatten()
        std = torch.as_tensor(normalizer.std, dtype=torch.float32, device=self.device).flatten()
        if mean.numel() != std.numel() or mean.numel() != 5:
            raise ValueError(
                f"Expected five channel normalization values, got mean={mean.shape}, std={std.shape}"
            )
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Channel standard deviations must be finite and positive")
        return mean, std

    def _channel_scale(self, value: torch.Tensor) -> torch.Tensor:
        """Broadcast per-channel statistics over [B,T,C,H,W] or [B,C,H,W]."""
        if value.dim() == 5:
            return self.channel_std.view(1, 1, -1, 1, 1)
        if value.dim() == 4:
            return self.channel_std.view(1, -1, 1, 1)
        raise ValueError(f"Expected a 4D or 5D tensor, got shape {value.shape}")

    def _broadcast_channels(self, values: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Broadcast channel statistics over a 4D or 5D field."""
        if value.dim() == 5:
            return values.view(1, 1, -1, 1, 1)
        if value.dim() == 4:
            return values.view(1, -1, 1, 1)
        raise ValueError(f"Expected a 4D or 5D tensor, got shape {value.shape}")

    def _get_gradient_statistics(self, gradient: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute detached per-channel mean and standard deviation of a gradient field."""
        channel_dim = 2 if gradient.dim() == 5 else 1 if gradient.dim() == 4 else None
        if channel_dim is None:
            raise ValueError(f"Expected a 4D or 5D gradient tensor, got shape {gradient.shape}")

        reduce_dims = tuple(dim for dim in range(gradient.dim()) if dim != channel_dim)
        with torch.no_grad():
            mean = gradient.mean(dim=reduce_dims)
            std = gradient.std(dim=reduce_dims, unbiased=False).clamp_min(torch.finfo(gradient.dtype).eps)
        return mean.detach(), std.detach()

    def _channel_offset(self, value: torch.Tensor) -> torch.Tensor:
        """Broadcast per-channel means over [B,T,C,H,W] or [B,C,H,W]."""
        if value.dim() == 5:
            return self.channel_mean.view(1, 1, -1, 1, 1)
        if value.dim() == 4:
            return self.channel_mean.view(1, -1, 1, 1)
        raise ValueError(f"Expected a 4D or 5D tensor, got shape {value.shape}")

    def predict_update(
        self,
        x: torch.Tensor,
        gradient_mean: Optional[torch.Tensor] = None,
        gradient_std: Optional[torch.Tensor] = None,
        target_gradient: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict IC update using S(θ, x).

        Args:
            x: Initial condition tensor, shape:
               - [C, H, W] -> output [C, H, W]
               - [B, C, H, W] -> output [B, C, H, W]
               - [B, T, C, H, W] -> output [B, T, C, H, W]

        Returns:
            update: Predicted IC update with same shape as input
        """
        x_in = x
        was_single = False
        if x_in.dim() == 3:
            x_in = x_in.unsqueeze(0)
            was_single = True

        # NetworkS receives ICs in the forward model's standardized channel
        # space and predicts the standardized target-gradient field.
        x_standardized = (x_in - self._channel_offset(x_in)) / self._channel_scale(x_in)
        predicted_gradient_standardized = self.network_s(x_standardized)
        if target_gradient is not None and (gradient_mean is None or gradient_std is None):
            gradient_mean, gradient_std = self._get_gradient_statistics(
                target_gradient.to(self.device).detach()
            )
        if gradient_mean is None or gradient_std is None:
            raise ValueError("gradient_mean and gradient_std are required to destandardize the prediction")
        update = (
            predicted_gradient_standardized * self._broadcast_channels(gradient_std, predicted_gradient_standardized)
            + self._broadcast_channels(gradient_mean, predicted_gradient_standardized)
        )

        # Apply ocean mask: self.ocean_mask may be [C, H, W] or [1, C, H, W]
        if update.dim() == 5:
            # [B, T, C, H, W] -> need mask shape [1, 1, C, H, W]
            if self.ocean_mask.dim() == 3:
                # [C, H, W] -> [1, 1, C, H, W]
                mask = self.ocean_mask.view(1, 1, *self.ocean_mask.shape)
            elif self.ocean_mask.dim() == 4:
                # [1, C, H, W] -> [1, 1, C, H, W]
                mask = self.ocean_mask.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected ocean_mask shape: {self.ocean_mask.shape}")
        elif update.dim() == 4:
            # [B, C, H, W] -> need mask shape [1, C, H, W]
            if self.ocean_mask.dim() == 3:
                # [C, H, W] -> [1, C, H, W]
                mask = self.ocean_mask.unsqueeze(0)
            elif self.ocean_mask.dim() == 4:
                # [1, C, H, W] -> use as-is
                mask = self.ocean_mask
            else:
                raise ValueError(f"Unexpected ocean_mask shape: {self.ocean_mask.shape}")
        else:
            raise ValueError(f"Unexpected update tensor shape: {update.shape}")

        update = update * mask

        if was_single:
            return update.squeeze(0)
        return update

    def get_diagnostics(self) -> Dict[str, float]:
        """Return accumulated diagnostics from meta-learner."""
        if not self.meta_loss_history:
            return {}

        return {
            "mean_L_meta": float(np.mean(self.meta_loss_history)),
            "num_updates": len(self.update_history),
        }
