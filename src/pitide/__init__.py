"""
Physics-Informed Time-Series Dense Encoder (PI-TiDE)
=====================================================

A physics-informed extension of the Time-Series Dense Encoder (TiDE)
for electricity load demand forecasting with guaranteed physical
constraint compliance.

Key Components:
- PITiDE: Main model class
- DegreeDayPhysics: Physics constraints (HDD/CDD, ramp rate, capacity envelope)
- PhysicsInformedLoss: Differentiable physics penalties
- train_pi_tide: Training loop with AMP and dynamic loss weighting

Example:
    >>> from pitide import PITiDE, train_pi_tide, DegreeDayPhysics
    >>> model = PITiDE(lookback=72, horizon=24, n_covariates=3, temp_idx=0)
    >>> model, _, _, metrics = train_pi_tide(df, use_physics=True, cooling_only=True)
"""

from .pitide import PITiDE
from .physics import DegreeDayPhysics
from .loss import physics_informed_loss
from .training import train_pi_tide
from .data import load_bangladesh_data, DemandWindowDataset, chronological_split, create_covariate_list
from .evaluation import evaluate_real_units, compute_violation_rates, run_ablation
from .hypertune import hypertune_pi_tide, build_grid, SEARCH_SPACE

__version__ = "1.0.0"
__author__ = "Physics-Informed TiDE Authors"
__email__ = "your.email@institution.edu"

__all__ = [
    "PITiDE",
    "DegreeDayPhysics",
    "physics_informed_loss",
    "train_pi_tide",
    "load_bangladesh_data",
    "DemandWindowDataset",
    "chronological_split",
    "create_covariate_list",
    "evaluate_real_units",
    "compute_violation_rates",
    "run_ablation",
    "hypertune_pi_tide",
    "build_grid",
    "SEARCH_SPACE",
]