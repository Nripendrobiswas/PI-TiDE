# Physics-Informed Time-Series Dense Encoder (PI-TiDE)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A physics-informed extension of the [Time-Series Dense Encoder (TiDE)](https://arxiv.org/abs/2304.08424) for electricity load demand forecasting that guarantees **zero physical constraint violations** while maintaining competitive point-forecast accuracy.

## Key Features

- **Physics-Informed Loss**: Differentiable ramp-rate limits, non-negativity, and temperature-dependent capacity envelope constraints
- **Dynamic Loss Weighting**: Adaptive gradient-norm balancing prevents over-constraining during training
- **Operational Validity**: Zero physical violations on Bangladesh Power Grid data (2016-2024)
- **Scalable Training**: Automatic Mixed Precision (AMP) + gradient accumulation (1.8× speedup, 40% memory reduction)
- **Hyperparameter Optimization**: Grid search over Table 7 ranges with fair baseline/PI-TiDE comparison
- **Comprehensive Ablation**: Systematic isolation of each physics component's contribution

## Results on Bangladesh Power Grid Data

| Model | MAE (MW) | MAPE (%) | RMSE (MW) | Physical Violations |
|-------|----------|----------|-----------|---------------------|
| Standard TiDE | 0.815 | 2.43 | 18.95 | 21.4% |
| **PI-TiDE (Ours)** | **0.842** | **2.36** | **18.34** | **0.0%** |

## Installation

```bash
git clone https://github.com/yourusername/PI-TiDE.git
cd PI-TiDE
pip install -r requirements.txt
```

## Quick Start

```python
from src.pitide import PITiDE, train_pi_tide
from src.data import load_bangladesh_data

# Load data
df = load_bangladesh_data("data/Kaggle_input_BangladeshData_2016_2024.csv")

# Train PI-TiDE
model, _, _, metrics = train_pi_tide(
    df,
    target_col="demand",
    temp_col="temp",
    covariate_cols=["temp", "humidity", "surface_pressure"],
    lookback=72,
    horizon=24,
    use_physics=True,
    cooling_only=True,           # Bangladesh: monotonic demand vs temperature
    physical_max_ramp=500.0,     # MW/h from grid specs
    lambda_env=0.02,
    epochs=50,
    patience=5
)
print(f"Test MAPE: {metrics['MAPE_%']:.2f}%")
```

## Repository Structure

```
PI-TiDE/
├── src/
│   ├── __init__.py
│   ├── pitide.py          # PITiDE model definition
│   ├── physics.py         # Physics constraints (DegreeDayPhysics)
│   ├── loss.py            # Physics-informed loss functions
│   ├── training.py        # Training loop with AMP + dynamic weighting
│   ├── data.py            # Data loading & preprocessing
│   ├── hypertune.py       # Grid search hyperparameter tuning
│   └── evaluation.py      # Metrics & evaluation utilities
├── notebooks/
│   └── pi_tide.ipynb      # Full experimental notebook
├── experiments/
│   ├── run_baseline.py    # Standard TiDE baseline
│   ├── run_pitide.py      # PI-TiDE with physics
│   └── run_ablation.py    # Systematic ablation study
├── data/                  # Place data files here
├── checkpoints/           # Model checkpoints (auto-created)
├── results/               # Experiment outputs (auto-created)
├── requirements.txt
├── setup.py
├── pyproject.toml
├── LICENSE
├── CITATION.cff
└── README.md
```

## Physics Constraints

| Constraint | Formula | Description |
|------------|---------|-------------|
| Ramp Rate | `ReLU(|ŷₕ - ŷₕ₋₁| - R_max)²` | Limits hourly demand changes |
| Non-Negativity | `ReLU(-ŷ)²` | Demand cannot be negative |
| Capacity Envelope | `ReLU(ŷ - P_max(T))²` | Temperature-dependent max demand |

## Dynamic Loss Weighting

Adaptive weights based on gradient magnitudes:
```
λ_k(t) = λ_k(0) × ‖∇L_data‖ / ‖∇L_k‖
```
Prevents physics penalties from dominating early training.

## Hyperparameter Tuning

Grid search over Table 7 ranges (25,920 configurations):

| Parameter | Range |
|-----------|-------|
| Hidden Size | [256, 512, 1024] |
| Encoder Layers | [1, 2, 3] |
| Decoder Layers | [1, 2, 3] |
| Decoder Output Dim | [4, 8, 16, 32] |
| Temporal Decoder Hidden | [32, 64, 128] |
| Dropout | [0.0, 0.1, 0.2, 0.3, 0.5] |
| LayerNorm | [True, False] |
| Learning Rate (log) | [1e-5, 1e-4, 1e-3, 1e-2] |
| RevIN | [True, False] |

Run: `python experiments/run_hypertune.py`

## Citation

If you use this work, please cite:

```bibtex
@article{nasir2026pitide,
  title={Physics-Informed Time-Series Dense Encoder for Electricity Load Demand Forecasting},
  author={Nasir, M.D. and Hanif, M.F. and Hassan, M.T. and Malik, M.I. and Tahir, A. and Husnain, N.},
  journal={Clean Energy},
  year={2026},
  doi={10.1093/ce/zkag048}
}
```

## Related Work

- [TiDE: Time-Series Dense Encoder](https://arxiv.org/abs/2304.08424) (Das et al., 2023)
- [Physics-Guided Temporal Fusion Transformer](https://doi.org/10.1093/ce/zkag048) (Nasir et al., 2026)
- [PhysEmbedFormer](https://doi.org/10.1038/s41598-025-34874-8) (Yu et al., 2026)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.