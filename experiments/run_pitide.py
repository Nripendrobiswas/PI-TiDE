#!/usr/bin/env python3
"""
Run PI-TiDE (Physics ON) with Improved Configuration
=====================================================
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pitide import (
    load_bangladesh_data,
    create_covariate_list,
    train_pi_tide,
)


def main():
    # Data
    data_path = Path(__file__).parent.parent / "data" / "Kaggle_input_BangladeshData_2016_2024.csv"
    df = load_bangladesh_data(str(data_path))
    covariate_cols = create_covariate_list(include_humidity=True, include_pressure=True)

    # Training config - improved physics settings
    common_kwargs = dict(
        target_col="demand",
        temp_col="temp",
        covariate_cols=covariate_cols,
        lookback=72,
        horizon=24,
        epochs=50,
        patience=5,
        batch_size=512,
        lr=1e-3,
        hidden_dim=64,
        encoder_layers=1,
        decoder_layers=1,
        decoder_output_dim=8,
        temporal_decoder_hidden=32,
        dropout=0.1,
        use_layer_norm=True,
        use_revin=False,
        physical_max_ramp=500.0,
    )

    # Physics-specific config for Bangladesh (cooling-only, monotonic demand vs temp)
    physics_kwargs = dict(
        use_physics=True,
        lambda_sens=0.02,
        lambda_nonneg=0.01,
        lambda_ramp=0.01,
        lambda_env=0.02,
        cooling_only=True,
        physical_max_ramp=500.0,
        dynamic_weighting=True,
    )

    print("=== PI-TiDE (Physics ON: Cooling-Only + Envelope + Dynamic Weighting) ===")
    torch.manual_seed(0)
    np.random.seed(0)

    _, _, _, metrics = train_pi_tide(
        df,
        checkpoint_path="checkpoints/pi_tide_best.pt",
        **common_kwargs,
        **physics_kwargs
    )

    print(f"\nPI-TiDE Test Metrics: {metrics}")

    # Save metrics
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    pd.DataFrame([metrics]).to_csv(results_dir / "pitide_metrics.csv", index=False)
    print(f"Saved metrics to {results_dir / 'pitide_metrics.csv'}")


if __name__ == "__main__":
    main()