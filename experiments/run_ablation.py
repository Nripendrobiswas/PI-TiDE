#!/usr/bin/env python3
"""
Run Systematic Ablation Study
=============================

Reproduces manuscript Table 5: isolates each physics component.
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
    run_ablation,
)


def main():
    # Data
    data_path = Path(__file__).parent.parent / "data" / "Kaggle_input_BangladeshData_2016_2024.csv"
    df = load_bangladesh_data(str(data_path))
    covariate_cols = create_covariate_list(include_humidity=True, include_pressure=True)

    # Common training config
    common_kwargs = dict(
        target_col="demand",
        temp_col="temp",
        covariate_cols=covariate_cols,
        lookback=72,
        horizon=24,
        epochs=15,           # Shorter for ablation
        patience=3,
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
        cooling_only=True,
    )

    # Ablation configurations matching manuscript Table 5
    ablation_configs = [
        ("TiDE_no_physics",          dict(use_physics=False, lambda_sens=0, lambda_nonneg=0, lambda_ramp=0, lambda_env=0)),
        ("Ramp_only",                dict(use_physics=True,  lambda_sens=0, lambda_nonneg=0, lambda_ramp=0.01, lambda_env=0)),
        ("Nonneg_only",              dict(use_physics=True,  lambda_sens=0, lambda_nonneg=0.01, lambda_ramp=0, lambda_env=0)),
        ("Envelope_only",            dict(use_physics=True,  lambda_sens=0, lambda_nonneg=0, lambda_ramp=0, lambda_env=0.02)),
        ("All_static_lambda",        dict(use_physics=True,  lambda_sens=0.1, lambda_nonneg=0.1, lambda_ramp=0.1, lambda_env=0.1)),
        ("PI_TiDE_dynamic_lambda",   dict(use_physics=True,  lambda_sens=0.02, lambda_nonneg=0.01, lambda_ramp=0.01, lambda_env=0.02)),
    ]

    print("=== Systematic Physics Ablation Study ===")
    print(f"Configurations: {len(ablation_configs)}")
    print()

    results_df = run_ablation(
        model_factory=lambda: None,  # train_pi_tide creates model internally
        train_fn=train_pi_tide,
        df=df,
        configs=ablation_configs,
        common_kwargs=common_kwargs,
    )

    print("\n=== Ablation Summary ===")
    print(results_df.to_string(index=False))

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_df.to_csv(results_dir / "ablation_results.csv", index=False)
    print(f"\nSaved to {results_dir / 'ablation_results.csv'}")

    # Also save as markdown for paper
    with open(results_dir / "ablation_table.md", "w") as f:
        f.write(results_df.to_markdown(index=False))
    print(f"Saved markdown table to {results_dir / 'ablation_table.md'}")


if __name__ == "__main__":
    main()