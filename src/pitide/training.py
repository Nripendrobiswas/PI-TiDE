"""
Training Loop for PI-TiDE
=========================

Features:
- Automatic Mixed Precision (AMP)
- Gradient accumulation
- Dynamic loss weighting (GradNorm-style)
- Early stopping
- Checkpointing
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict, Any, List

from .pitide import PITiDE
from .physics import DegreeDayPhysics, estimate_balance_temp_piecewise
from .loss import physics_informed_loss, compute_violation_rates


class DemandWindowDataset(Dataset):
    """Sliding window dataset for demand + covariates."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        covariate_cols: List[str],
        lookback: int,
        horizon: int,
    ):
        self.target = df[target_col].values.astype(np.float32)
        self.covariates = df[covariate_cols].values.astype(np.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.n = len(df) - lookback - horizon + 1
        if self.n <= 0:
            raise ValueError("DataFrame too short for given lookback/horizon.")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_target = self.target[idx: idx + self.lookback]
        cov_window = self.covariates[idx: idx + self.lookback + self.horizon]
        future_target = self.target[idx + self.lookback: idx + self.lookback + self.horizon]
        return (
            torch.from_numpy(past_target),
            torch.from_numpy(cov_window),
            torch.from_numpy(future_target),
        )


def chronological_split(
    dataset: Dataset,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Time-ordered split: train / val / test."""
    n = len(dataset)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_ds = torch.utils.data.Subset(dataset, range(0, n_train))
    val_ds = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
    test_ds = torch.utils.data.Subset(dataset, range(n_train + n_val, n))
    return train_ds, val_ds, test_ds


@torch.no_grad()
def evaluate_real_units(
    model: nn.Module,
    loader: DataLoader,
    target_scaler: StandardScaler,
    device: str,
) -> Dict[str, float]:
    """Compute MAE, MAPE, RMSE in original demand units."""
    model.eval()
    all_preds, all_true = [], []
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


def train_pi_tide(
    df: pd.DataFrame,
    target_col: str = "demand",
    temp_col: str = "temp",
    covariate_cols: Optional[List[str]] = None,
    lookback: int = 168,
    horizon: int = 24,
    balance_temp: Optional[float] = None,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    # Architecture
    hidden_dim: int = 128,
    encoder_layers: int = 2,
    decoder_layers: int = 2,
    decoder_output_dim: int = 16,
    temporal_decoder_hidden: Optional[int] = None,
    dropout: float = 0.1,
    use_layer_norm: bool = True,
    use_revin: bool = False,
    # Physics
    use_physics: bool = True,
    lambda_sens: float = 0.1,
    lambda_nonneg: float = 0.1,
    lambda_ramp: float = 0.1,
    lambda_env: float = 0.1,
    cooling_only: bool = False,
    physical_max_ramp: Optional[float] = None,
    # Dynamic loss weighting
    dynamic_weighting: bool = True,
    # Training
    patience: int = 5,
    checkpoint_path: str = "pi_tide_best.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    return_val_loss: bool = False,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
) -> Tuple[nn.Module, StandardScaler, StandardScaler, Dict[str, float], Optional[float]]:
    """
    Train PI-TiDE with physics-informed loss.

    Args:
        df: DataFrame with demand, temperature, and covariates
        target_col: Target column name
        temp_col: Temperature column name
        covariate_cols: List of covariate columns
        lookback: Input window length
        horizon: Forecast horizon
        balance_temp: Pre-specified balance temp (None -> piecewise estimate)
        batch_size: Training batch size
        epochs: Max epochs
        lr: Learning rate
        hidden_dim: Model width
        encoder_layers: Encoder depth
        decoder_layers: Decoder depth
        decoder_output_dim: Decoder output width per step
        temporal_decoder_hidden: Temporal decoder hidden size
        dropout: Dropout rate
        use_layer_norm: Use LayerNorm in residual blocks
        use_revin: Use Reversible Instance Norm
        use_physics: Enable physics-informed loss
        lambda_sens: Sensitivity sign weight
        lambda_nonneg: Non-negativity weight
        lambda_ramp: Ramp rate weight
        lambda_env: Capacity envelope weight
        cooling_only: Monotonic cooling-only regime
        physical_max_ramp: Physical ramp limit (MW/h), overrides percentile
        dynamic_weighting: Use GradNorm-style adaptive weights
        patience: Early stopping patience
        checkpoint_path: Path to save best model
        device: Training device
        verbose: Print progress
        return_val_loss: Return best validation loss
        grad_accum_steps: Gradient accumulation steps
        use_amp: Use Automatic Mixed Precision

    Returns:
        model, target_scaler, cov_scaler, test_metrics, [best_val_loss]
    """
    if covariate_cols is None:
        covariate_cols = [temp_col]
    temp_idx = covariate_cols.index(temp_col)

    # Estimate balance temperature if not provided
    if balance_temp is None:
        balance_temp = estimate_balance_temp_piecewise(df, target_col, temp_col)
        if verbose:
            print(f"Estimated balance_temp (piecewise): {balance_temp:.2f}°C")

    # Prepare data
    df = df.copy()
    n_total = len(df)
    n_train_rows = int(0.8 * n_total)

    target_scaler = StandardScaler().fit(df[[target_col]].iloc[:n_train_rows])
    cov_scaler = StandardScaler().fit(df[covariate_cols].iloc[:n_train_rows])

    df[target_col] = target_scaler.transform(df[[target_col]]).ravel()
    df[covariate_cols] = cov_scaler.transform(df[covariate_cols])

    temp_mean = cov_scaler.mean_[temp_idx]
    temp_std = cov_scaler.scale_[temp_idx]
    scaled_balance_temp = (balance_temp - temp_mean) / temp_std

    # Create datasets
    dataset = DemandWindowDataset(df, target_col, covariate_cols, lookback, horizon)
    train_ds, val_ds, test_ds = chronological_split(dataset, train_frac=0.8, val_frac=0.1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Physical max ramp -> scaled
    if physical_max_ramp is not None:
        max_ramp = physical_max_ramp / target_scaler.scale_[0]
    else:
        raw_train_target = df[target_col].values[:n_train_rows]
        max_ramp = float(np.percentile(np.abs(np.diff(raw_train_target)), 99))

    # Model
    model = PITiDE(
        lookback=lookback,
        horizon=horizon,
        n_covariates=len(covariate_cols),
        temp_idx=temp_idx,
        hidden_dim=hidden_dim,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        decoder_output_dim=decoder_output_dim,
        temporal_decoder_hidden=temporal_decoder_hidden,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
        use_revin=use_revin,
    ).to(device)

    physics = DegreeDayPhysics(balance_temp=scaled_balance_temp, cooling_only=cooling_only)

    # Fit temperature-power envelope on training data
    train_df = df.iloc[:n_train_rows].copy()
    train_df[target_col] = target_scaler.inverse_transform(train_df[[target_col]]).ravel()
    temp_scaled = train_df[[temp_col]].values
    temp_orig = cov_scaler.inverse_transform(
        np.hstack([temp_scaled, np.zeros((len(temp_scaled), len(covariate_cols) - 1))])
    )[:, 0]
    train_df[temp_col] = temp_orig
    physics.fit_envelope(train_df, temp_col=temp_col, target_col=target_col)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=use_amp)

    # Initial lambda weights for dynamic weighting
    lambda_init = {
        "sens": lambda_sens,
        "nonneg": lambda_nonneg,
        "ramp": lambda_ramp,
        "env": lambda_env,
    }
    lambda_current = lambda_init.copy()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        optimizer.zero_grad()

        for step, (past_target, covariates, future_target) in enumerate(train_loader):
            past_target = past_target.to(device)
            covariates = covariates.to(device)
            future_target = future_target.to(device)

            with autocast(enabled=use_amp):
                if use_physics:
                    covariates.requires_grad_(True)
                preds = model(past_target, covariates)

                data_loss = F.mse_loss(preds, future_target)

                if use_physics:
                    phys_loss, _ = physics_informed_loss(
                        model, past_target, covariates, preds, physics, max_ramp,
                        lambda_sens=lambda_current["sens"],
                        lambda_nonneg=lambda_current["nonneg"],
                        lambda_ramp=lambda_current["ramp"],
                        lambda_env=lambda_current["env"],
                    )
                else:
                    phys_loss = torch.tensor(0.0, device=device)

                loss = (data_loss + phys_loss) / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_data_loss += data_loss.item() * past_target.size(0)
            epoch_phys_loss += float(phys_loss.detach()) * past_target.size(0)

        epoch_data_loss /= len(train_ds)
        epoch_phys_loss /= len(train_ds)

        # --- Dynamic loss weighting (GradNorm-style) ---
        if dynamic_weighting and use_physics:
            with torch.no_grad():
                # Compute gradient norms for each loss component
                grad_norms = {}
                for name, weight in [("sens", lambda_sens), ("nonneg", lambda_nonneg),
                                      ("ramp", lambda_ramp), ("env", lambda_env)]:
                    # Use a small batch to estimate gradient norm
                    sample_past = past_target[:min(32, len(past_target))].detach()
                    sample_cov = covariates[:min(32, len(covariates))].detach().requires_grad_(True)
                    sample_pred = model(sample_past, sample_cov)
                    sample_phys_loss, _ = physics_informed_loss(
                        model, sample_past, sample_cov, sample_pred, physics, max_ramp,
                        lambda_sens=lambda_sens if name == "sens" else 0,
                        lambda_nonneg=lambda_nonneg if name == "nonneg" else 0,
                        lambda_ramp=lambda_ramp if name == "ramp" else 0,
                        lambda_env=lambda_env if name == "env" else 0,
                    )
                    grad = torch.autograd.grad(sample_phys_loss, model.parameters(),
                                                retain_graph=True, allow_unused=True)
                    total_norm = sum(g.norm(2).item() ** 2 for g in grad if g is not None) ** 0.5
                    grad_norms[name] = total_norm + 1e-8

                data_grad = torch.autograd.grad(data_loss, model.parameters(),
                                                 retain_graph=False, allow_unused=True)
                data_norm = sum(g.norm(2).item() ** 2 for g in data_grad if g is not None) ** 0.5

                # Update lambda weights
                for name in lambda_current:
                    lambda_current[name] = lambda_init[name] * (data_norm / grad_norms[name])

        # --- Validation ---
        model.eval()
        val_loss_total = 0.0
        viol_counts = {"sens": 0, "nonneg": 0, "ramp": 0, "env": 0, "total": 0}
        viol_totals = {"sens": 0, "nonneg": 0, "ramp": 0, "env": 0, "total": 0}

        # First pass: compute val loss (no grad)
        with torch.no_grad():
            for past_target, covariates, future_target in val_loader:
                past_target = past_target.to(device)
                covariates = covariates.to(device)
                future_target = future_target.to(device)
                preds = model(past_target, covariates)
                val_loss_total += F.mse_loss(preds, future_target).item() * past_target.size(0)

        # Second pass: compute violation rates (requires grad)
        if use_physics:
            for past_target, covariates, future_target in val_loader:
                past_target = past_target.to(device)
                covariates = covariates.to(device).requires_grad_(True)
                future_target = future_target.to(device)
                preds = model(past_target, covariates)

                viol = compute_violation_rates(model, past_target, covariates, preds, physics, max_ramp)
                for k in viol_counts:
                    if k in viol:
                        viol_counts[k] += viol[k] * past_target.size(0)
                    viol_totals[k] += past_target.size(0)

        val_loss = val_loss_total / len(val_ds)

        if verbose:
            viol_rates = {k: viol_counts[k] / max(viol_totals[k], 1) for k in viol_counts}
            print(
                f"epoch {epoch:3d} | data_loss {epoch_data_loss:.4f} | phys_loss {epoch_phys_loss:.4f} "
                f"| val_loss {val_loss:.4f} | sens_viol {viol_rates['sens']:.3f} "
                f"ramp_viol {viol_rates['ramp']:.3f} env_viol {viol_rates['env']:.3f}"
            )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "lookback": lookback, "horizon": horizon,
                    "n_covariates": len(covariate_cols), "temp_idx": temp_idx,
                    "hidden_dim": hidden_dim, "encoder_layers": encoder_layers,
                    "decoder_layers": decoder_layers, "decoder_output_dim": decoder_output_dim,
                    "temporal_decoder_hidden": temporal_decoder_hidden,
                    "dropout": dropout, "use_layer_norm": use_layer_norm,
                    "use_revin": use_revin, "lr": lr,
                },
                "lambda_weights": lambda_current,
            }, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # Load best model and evaluate
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    test_metrics = evaluate_real_units(model, test_loader, target_scaler, device)

    if verbose:
        print(f"Test set (original units): {test_metrics}")

    if return_val_loss:
        return model, target_scaler, cov_scaler, test_metrics, best_val_loss
    return model, target_scaler, cov_scaler, test_metrics