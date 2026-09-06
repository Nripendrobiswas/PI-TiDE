"""
Evaluation Metrics and Analysis
===============================

Comprehensive metrics for physics-informed forecasting:
- Standard point-forecast metrics
- Physical constraint violation rates
- Per-regime analysis
- Ablation utilities
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_real_units(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    target_scaler,
    device: str,
) -> Dict[str, float]:
    """
    Compute MAE, MAPE, RMSE in original demand units.

    Args:
        model: Trained model
        loader: Test DataLoader
        target_scaler: Fitted StandardScaler for target
        device: 'cuda' or 'cpu'

    Returns:
        Dict with MAPE_%, RMSE, MAE
    """
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for past_target, covariates, future_target in loader:
            past_target = past_target.to(device)
            covariates = covariates.to(device)
            preds = model(past_target, covariates)
            all_preds.append(preds.cpu().numpy())
            all_true.append(future_target.numpy())

    preds = np.concatenate(all_preds, axis=0).reshape(-1, 1)
    true = np.concatenate(all_true, axis=0).reshape(-1, 1)

    preds_real = target_scaler.inverse_transform(preds).ravel()
    true_real = target_scaler.inverse_transform(true).ravel()

    mape = float(np.mean(np.abs((true_real - preds_real) / (true_real + 1e-6))) * 100)
    rmse = float(np.sqrt(np.mean((true_real - preds_real) ** 2)))
    mae = float(np.mean(np.abs(true_real - preds_real)))
    return {"MAPE_%": mape, "RMSE": rmse, "MAE": mae}


def compute_violation_rates(
    model: torch.nn.Module,
    past_target: torch.Tensor,
    covariates: torch.Tensor,
    predictions: torch.Tensor,
    physics,
    max_ramp: float,
) -> Dict[str, float]:
    """
    Compute physical violation rates.

    Note: Caller must ensure gradients are enabled (covariates.requires_grad_(True))
    and this is called outside torch.no_grad() context.
    """
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

    total_viol = float(sens_viol > 0 or nonneg_viol > 0 or ramp_viol > 0 or env_viol > 0)

    return {
        "sensitivity_violation_rate": sens_viol,
        "nonneg_violation_rate": nonneg_viol,
        "ramp_violation_rate": ramp_viol,
        "envelope_violation_rate": env_viol,
        "total_violation_rate": total_viol,
    }


def per_regime_analysis(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    physics,
    max_ramp: float,
    device: str,
) -> Dict[str, Dict[str, float]]:
    """
    Analyze violations by temperature regime (heating vs cooling).

    Returns:
        Dict with 'cold' and 'hot' sub-dicts containing violation rates
    """
    model.eval()
    cold_viols = {"sens": 0, "nonneg": 0, "ramp": 0, "env": 0, "total": 0}
    hot_viols = {"sens": 0, "nonneg": 0, "ramp": 0, "env": 0, "total": 0}
    cold_count = 0
    hot_count = 0

    with torch.no_grad():
        for past_target, covariates, future_target in loader:
            past_target = past_target.to(device)
            covariates = covariates.to(device).requires_grad_(True)
            preds = model(past_target, covariates)

            future_temp = covariates[:, model.lookback:, model.temp_idx]
            cold_mask = future_temp < physics.T_b
            hot_mask = future_temp >= physics.T_b

            if cold_mask.any():
                grad_outputs = torch.ones_like(preds)
                d_pred_d_cov = torch.autograd.grad(preds, covariates, grad_outputs=grad_outputs)[0]
                d_pred_d_temp = d_pred_d_cov[:, model.lookback:, model.temp_idx]
                expected_sign = physics.expected_sensitivity_sign(future_temp)

                cold_viols["sens"] += ((expected_sign * d_pred_d_temp < 0).float()[cold_mask].sum().item())
                cold_viols["nonneg"] += ((preds < 0).float()[cold_mask].sum().item())
                diffs = preds[:, 1:] - preds[:, :-1]
                cold_viols["ramp"] += ((diffs.abs() > max_ramp).float()[cold_mask].sum().item())
                if hasattr(physics, "max_power_at_temp") and physics.envelope_temps is not None:
                    max_power = physics.max_power_at_temp(future_temp)
                    cold_viols["env"] += ((preds > max_power).float()[cold_mask].sum().item())
                cold_count += cold_mask.sum().item()

            if hot_mask.any():
                hot_viols["sens"] += ((expected_sign * d_pred_d_temp < 0).float()[hot_mask].sum().item())
                hot_viols["nonneg"] += ((preds < 0).float()[hot_mask].sum().item())
                diffs = preds[:, 1:] - preds[:, :-1]
                hot_viols["ramp"] += ((diffs.abs() > max_ramp).float()[hot_mask].sum().item())
                if hasattr(physics, "max_power_at_temp") and physics.envelope_temps is not None:
                    max_power = physics.max_power_at_temp(future_temp)
                    hot_viols["env"] += ((preds > max_power).float()[hot_mask].sum().item())
                hot_count += hot_mask.sum().item()

    return {
        "cold": {k: v / max(cold_count, 1) for k, v in cold_viols.items()},
        "hot": {k: v / max(hot_count, 1) for k, v in hot_viols.items()},
    }


def run_ablation(
    model_factory,
    train_fn,
    df,
    configs: List[Tuple[str, dict]],
    common_kwargs: dict,
) -> pd.DataFrame:
    """
    Run systematic ablation study.

    Args:
        model_factory: Function returning new model instance
        train_fn: Training function
        df: DataFrame
        configs: List of (name, physics_kwargs) tuples
        common_kwargs: Shared training arguments

    Returns:
        DataFrame with results
    """
    import pandas as pd
    results = []

    for name, phys_kwargs in configs:
        print(f"Running ablation: {name}...")
        model = model_factory()
        _, _, _, test_metrics, val_loss = train_fn(
            df, checkpoint_path=f"ablation_{name}.pt",
            return_val_loss=True, verbose=False,
            **common_kwargs, **phys_kwargs
        )
        results.append({
            "Configuration": name,
            "MAPE_%": f"{test_metrics['MAPE_%']:.2f}",
            "RMSE": f"{test_metrics['RMSE']:.2f}",
            "MAE": f"{test_metrics.get('MAE', 0):.2f}",
            "Val_Loss": f"{val_loss:.4f}",
        })

    return pd.DataFrame(results)