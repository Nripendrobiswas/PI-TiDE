#!/usr/bin/env python3
"""
Run Hyperparameter Grid Search (Table 7)
========================================

Grid search over Table 7 ranges for both Baseline TiDE and PI-TiDE.
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
    hypertune_pi_tide,
)


def main():
    # Data
    data_path = Path(__file__).parent.parent / "data" / "Kaggle_input_BangladeshData_2016_2024.csv"
    df = load_bangladesh_data(str(data_path))
    covariate_cols = create_covariate_list(include_humidity=True, include_pressure=True)

    # Hypertuning config
    hypertune_kwargs = dict(
        target_col="demand",
        temp_col="temp",
        covariate_cols=covariate_cols,
        lookback=72,
        horizon=24,
        max_trials=20,        # Set to None for full grid (25,920 runs)
        trial_epochs=15,
        trial_patience=3,
        batch_size=512,
        lr=1e-3,
        seed=0,
    )

    print("=== Hyperparameter Grid Search (Table 7) ===")
    print(f"Full grid size: 25,920 configurations per model")
    print(f"Running first {hypertune_kwargs['max_trials']} trials per model\n")

    # Baseline TiDE (Physics OFF)
    print("=" * 60)
    print("Baseline TiDE Hypertune (Physics OFF)")
    print("=" * 60)
    best_baseline, results_baseline = hypertune_pi_tide(
        df,
        use_physics=False,
        checkpoint_dir="hypertune_ckpts_baseline",
        **hypertune_kwargs
    )

    print("\nBaseline Top-5:")
    print(results_baseline.head(5).to_string())

    # PI-TiDE (Physics ON)
    print("\n" + "=" * 60)
    print("PI-TiDE Hypertune (Physics ON)")
    print("=" * 60)
    best_pi, results_pi = hypertune_pi_tide(
        df,
        use_physics=True,
        lambda_sens=0.02,
        lambda_nonneg=0.01,
        lambda_ramp=0.01,
        lambda_env=0.02,
        cooling_only=True,
        physical_max_ramp=500.0,
        checkpoint_dir="hypertune_ckpts_pi",
        **hypertune_kwargs
    )

    print("\nPI-TiDE Top-5:")
    print(results_pi.head(5).to_string())

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_baseline.to_csv(results_dir / "hypertune_baseline.csv", index=False)
    results_pi.to_csv(results_dir / "hypertune_pi_tide.csv", index=False)

    # Save best configs
    pd.DataFrame([best_baseline]).to_csv(results_dir / "best_baseline_config.csv", index=False)
    pd.DataFrame([best_pi]).to_csv(results_dir / "best_pi_tide_config.csv", index=False)

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()