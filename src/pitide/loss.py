"""
Physics-Informed Loss Functions
===============================

Differentiable penalty terms for physical constraint enforcement:
- Sensitivity sign consistency (HDD/CDD regime)
- Non-negativity
- Ramp rate limits
- Temperature-power capacity envelope
"""

import torch
import torch.nn.functional as F
from typing import Tuple
from .physics import DegreeDayPhysics


def physics_informed_loss(
    model: torch.nn.Module,
    past_target: torch.Tensor,
    covariates: torch.Tensor,
    predictions: torch.Tensor,
    physics: DegreeDayPhysics,
    max_ramp: float,
    lambda_sens: float = 0.1,
    lambda_nonneg: float = 0.1,
    lambda_ramp: float = 0.1,
    lambda_env: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute physics-informed loss components.

    Args:
        model: PITiDE model (for lookback/temp_idx attributes)
        past_target: (B, L) - historical demand
        covariates: (B, L+H, C) - covariates including temperature
        predictions: (B, H) - forecasted demand
        physics: DegreeDayPhysics instance
        max_ramp: Maximum allowed ramp (scaled units)
        lambda_sens: Sensitivity sign weight
        lambda_nonneg: Non-negativity weight
        lambda_ramp: Ramp rate weight
        lambda_env: Capacity envelope weight

    Returns:
        total_loss: Scalar tensor
        components: Dict with individual loss values
    """
    future_temp = covariates[:, model.lookback:, model.temp_idx]  # (B, H)

    # --- Sensitivity sign consistency ---
    # Gradient of predictions w.r.t. covariates
    grad_outputs = torch.ones_like(predictions)
    d_pred_d_cov = torch.autograd.grad(
        outputs=predictions,
        inputs=covariates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]  # (B, L+H, C)
    d_pred_d_temp = d_pred_d_cov[:, model.lookback:, model.temp_idx]  # (B, H)

    expected_sign = physics.expected_sensitivity_sign(future_temp)
    sensitivity_violation = F.relu(-expected_sign * d_pred_d_temp)
    sensitivity_loss = sensitivity_violation.mean()

    # --- Non-negativity ---
    nonneg_loss = F.relu(-predictions).pow(2).mean()

    # --- Ramp rate ---
    diffs = predictions[:, 1:] - predictions[:, :-1]
    ramp_violation = F.relu(diffs.abs() - max_ramp)
    ramp_loss = ramp_violation.pow(2).mean()

    # --- Temperature-power envelope ---
    if hasattr(physics, "max_power_at_temp") and physics.envelope_temps is not None:
        max_power = physics.max_power_at_temp(future_temp)
        envelope_violation = F.relu(predictions - max_power).pow(2).mean()
    else:
        envelope_violation = torch.tensor(0.0, device=predictions.device)

    # Total loss with dynamic weighting support
    total = (
        lambda_sens * sensitivity_loss +
        lambda_nonneg * nonneg_loss +
        lambda_ramp * ramp_loss +
        lambda_env * envelope_violation
    )

    components = {
        "sensitivity_loss": sensitivity_loss.item(),
        "nonneg_loss": nonneg_loss.item(),
        "ramp_loss": ramp_loss.item(),
        "envelope_loss": envelope_violation.item(),
    }
    return total, components


def compute_violation_rates(
    model: torch.nn.Module,
    past_target: torch.Tensor,
    covariates: torch.Tensor,
    predictions: torch.Tensor,
    physics: DegreeDayPhysics,
    max_ramp: float,
) -> dict:
    """
    Compute physical violation rates for monitoring (no gradients).

    Returns:
        Dict with violation rates for each constraint
    """
    with torch.no_grad():
        future_temp = covariates[:, model.lookback:, model.temp_idx]
        grad_outputs = torch.ones_like(predictions)
        d_pred_d_cov = torch.autograd.grad(
            outputs=predictions,
            inputs=covariates,
            grad_outputs=grad_outputs,
            retain_graph=False,
        )[0]
        d_pred_d_temp = d_pred_d_cov[:, model.lookback:, model.temp_idx]

        expected_sign = physics.expected_sensitivity_sign(future_temp)
        sens_viol = (expected_sign * d_pred_d_temp < 0).float().mean().item()
        nonneg_viol = (predictions < 0).float().mean().item()
        diffs = predictions[:, 1:] - predictions[:, :-1]
        ramp_viol = (diffs.abs() > max_ramp).float().mean().item()

        if hasattr(physics, "max_power_at_temp") and physics.envelope_temps is not None:
            max_power = physics.max_power_at_temp(future_temp)
            env_viol = (predictions > max_power).float().mean().item()
        else:
            env_viol = 0.0

        total_viol = (sens_viol > 0 or nonneg_viol > 0 or ramp_viol > 0 or env_viol > 0)

    return {
        "sensitivity_violation_rate": sens_viol,
        "nonneg_violation_rate": nonneg_viol,
        "ramp_violation_rate": ramp_viol,
        "envelope_violation_rate": env_viol,
        "total_violation_rate": float(total_viol),
    }