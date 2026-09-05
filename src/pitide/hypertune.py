"""
Hyperparameter Tuning for PI-TiDE
==================================

Grid search over Table 7 ranges with deterministic ordering.
Supports narrowing search space and early stopping per trial.
"""

import itertools
import random
import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any

# Table 7: Hyperparameter search ranges
SEARCH_SPACE = {
    "hiddenSize": [256, 512, 1024],
    "numEncoderLayers": [1, 2, 3],
    "numDecoderLayers": [1, 2, 3],
    "decoderOutputDim": [4, 8, 16, 32],
    "temporalDecoderHidden": [32, 64, 128],
    "dropoutLevel": [0.0, 0.1, 0.2, 0.3, 0.5],
    "layerNorm": [True, False],
    "learningRate": [1e-5, 1e-4, 1e-3, 1e-2],  # log-spaced grid
    "revIn": [True, False],
}

# Fixed key order for deterministic grid
GRID_KEYS = [
    "hiddenSize", "numEncoderLayers", "numDecoderLayers", "decoderOutputDim",
    "temporalDecoderHidden", "dropoutLevel", "layerNorm", "learningRate", "revIn"
]


def build_grid(search_space: Optional[Dict] = None) -> List[Dict]:
    """
    Generate Cartesian product of hyperparameter grid.

    Args:
        search_space: Optional subset of SEARCH_SPACE to narrow grid

    Returns:
        List of config dicts in deterministic order
    """
    ss = search_space if search_space is not None else SEARCH_SPACE
    keys = [k for k in GRID_KEYS if k in ss]
    return [dict(zip(keys, combo)) for combo in itertools.product(*(ss[k] for k in keys))]


def grid_size(search_space: Optional[Dict] = None) -> int:
    """Total number of grid points."""
    ss = search_space if search_space is not None else SEARCH_SPACE
    n = 1
    for k in GRID_KEYS:
        if k in ss:
            n *= len(ss[k])
    return n


def sample_config(rng: random.Random, search_space: Optional[Dict] = None) -> Dict:
    """
    Random sample from search space (for random search fallback).

    Args:
        rng: Random number generator
        search_space: Optional custom search space

    Returns:
        Sampled configuration dict
    """
    ss = search_space if search_space is not None else SEARCH_SPACE
    lo, hi = ss["learningRate"]
    log_lo, log_hi = np.log10(lo), np.log10(hi)
    return {
        "hiddenSize": rng.choice(ss["hiddenSize"]),
        "numEncoderLayers": rng.choice(ss["numEncoderLayers"]),
        "numDecoderLayers": rng.choice(ss["numDecoderLayers"]),
        "decoderOutputDim": rng.choice(ss["decoderOutputDim"]),
        "temporalDecoderHidden": rng.choice(ss["temporalDecoderHidden"]),
        "dropoutLevel": rng.choice(ss["dropoutLevel"]),
        "layerNorm": rng.choice(ss["layerNorm"]),
        "learningRate": float(10 ** rng.uniform(log_lo, log_hi)),
        "revIn": rng.choice(ss["revIn"]),
    }


def hypertune_pi_tide(
    df: pd.DataFrame,
    target_col: str = "demand",
    temp_col: str = "temp",
    covariate_cols: Optional[List[str]] = None,
    lookback: int = 72,
    horizon: int = 24,
    balance_temp: Optional[float] = None,
    search_space: Optional[Dict] = None,
    max_trials: Optional[int] = None,
    trial_epochs: int = 15,
    trial_patience: int = 3,
    batch_size: int = 64,
    use_physics: bool = True,
    lambda_sens: float = 0.1,
    lambda_nonneg: float = 0.1,
    lambda_ramp: float = 0.1,
    lambda_env: float = 0.1,
    cooling_only: bool = False,
    physical_max_ramp: Optional[float] = None,
    seed: int = 0,
    checkpoint_dir: str = "hypertune_ckpts",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Grid search hyperparameter tuning for PI-TiDE.

    Args:
        df: Training DataFrame
        target_col: Target column name
        temp_col: Temperature column name
        covariate_cols: Covariate columns
        lookback: Input window length
        horizon: Forecast horizon
        balance_temp: Fixed balance temperature (None -> estimate once)
        search_space: Custom search space (subset of SEARCH_SPACE)
        max_trials: Max grid points to evaluate (None = full grid)
        trial_epochs: Epochs per trial
        trial_patience: Early stopping patience per trial
        batch_size: Batch size
        use_physics: Enable physics loss
        lambda_sens: Sensitivity weight
        lambda_nonneg: Non-negativity weight
        lambda_ramp: Ramp rate weight
        lambda_env: Envelope weight
        cooling_only: Monotonic cooling regime
        physical_max_ramp: Physical ramp limit (MW/h)
        seed: Random seed
        checkpoint_dir: Directory for trial checkpoints
        device: Training device
        verbose: Print progress

    Returns:
        (best_config, results_dataframe)
    """
    import pandas as pd
    os.makedirs(checkpoint_dir, exist_ok=True)

    if covariate_cols is None:
        covariate_cols = [temp_col]
    if balance_temp is None:
        from .physics import estimate_balance_temp_piecewise
        balance_temp = estimate_balance_temp_piecewise(df, target_col, temp_col)
        if verbose:
            print(f"[hypertune] Fixed balance_temp={balance_temp:.2f} for all trials")

    # Import training function locally to avoid circular imports
    from .training import train_pi_tide

    grid = build_grid(search_space)
    total = len(grid)
    if max_trials is not None:
        grid = grid[:max_trials]
    if verbose:
        print(f"[hypertune] Grid search: running {len(grid)}/{total} configurations")

    records = []
    for trial, cfg in enumerate(grid):
        torch.manual_seed(seed + trial)
        np.random.seed(seed + trial)
        ckpt = os.path.join(checkpoint_dir, f"trial_{trial:03d}.pt")

        _, _, _, test_metrics, val_loss = train_pi_tide(
            df,
            target_col=target_col, temp_col=temp_col, covariate_cols=covariate_cols,
            lookback=lookback, horizon=horizon, balance_temp=balance_temp,
            batch_size=batch_size, epochs=trial_epochs, patience=trial_patience,
            lr=cfg["learningRate"],
            hidden_dim=cfg["hiddenSize"],
            encoder_layers=cfg["numEncoderLayers"],
            decoder_layers=cfg["numDecoderLayers"],
            decoder_output_dim=cfg["decoderOutputDim"],
            temporal_decoder_hidden=cfg["temporalDecoderHidden"],
            dropout=cfg["dropoutLevel"],
            use_layer_norm=cfg["layerNorm"],
            use_revin=cfg["revIn"],
            use_physics=use_physics,
            lambda_sens=lambda_sens, lambda_nonneg=lambda_nonneg,
            lambda_ramp=lambda_ramp, lambda_env=lambda_env,
            cooling_only=cooling_only,
            physical_max_ramp=physical_max_ramp,
            checkpoint_path=ckpt,
            device=device, verbose=False, return_val_loss=True,
        )

        rec = {
            **cfg,
            "val_loss": float(val_loss),
            "test_MAPE": test_metrics["MAPE_%"],
            "test_RMSE": test_metrics["RMSE"],
            "test_MAE": test_metrics.get("MAE", 0.0),
            "checkpoint": ckpt,
        }
        records.append(rec)

        if verbose:
            print(
                f"trial {trial + 1}/{len(grid)} | val {val_loss:.4f} | "
                f"MAPE {test_metrics['MAPE_%']:.2f}% RMSE {test_metrics['RMSE']:.2f} | {cfg}"
            )

    results_df = pd.DataFrame(records).sort_values("val_loss", ignore_index=True)
    best = results_df.iloc[0].to_dict()

    if verbose:
        print("\nBest config (lowest val_loss):")
        for k, v in best.items():
            print(f"  {k}: {v}")

    return best, results_df


# For backward compatibility
__all__ = [
    "SEARCH_SPACE",
    "GRID_KEYS",
    "build_grid",
    "grid_size",
    "sample_config",
    "hypertune_pi_tide",
]