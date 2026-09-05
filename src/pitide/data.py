"""
Data Loading and Preprocessing
==============================

Utilities for loading Bangladesh Power Grid data and general
time-series preprocessing.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from torch.utils.data import Dataset
from .training import DemandWindowDataset, chronological_split


def load_bangladesh_data(
    filepath: str,
    target_col: str = "Demand",
    temp_col: str = "temperature",
    rename_cols: bool = True,
) -> pd.DataFrame:
    """
    Load Bangladesh Power Grid dataset (2016-2024).

    Args:
        filepath: Path to CSV file
        target_col: Original target column name
        temp_col: Original temperature column name
        rename_cols: Rename to standard 'demand'/'temp'

    Returns:
        DataFrame with columns: DateTime, demand, temp, humidity, surface_pressure, Generation
    """
    df = pd.read_csv(filepath, parse_dates=["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)

    if rename_cols:
        df = df.rename(columns={target_col: "demand", temp_col: "temp"})

    return df


def add_time_features(df: pd.DataFrame, datetime_col: str = "DateTime") -> pd.DataFrame:
    """Add cyclic time features (hour, day, month)."""
    df = df.copy()
    dt = df[datetime_col]
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df["day_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    return df


def filter_outliers(
    df: pd.DataFrame,
    target_col: str = "demand",
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.999,
) -> pd.DataFrame:
    """Remove extreme outliers from target column."""
    lo = df[target_col].quantile(lower_quantile)
    hi = df[target_col].quantile(upper_quantile)
    return df[(df[target_col] >= lo) & (df[target_col] <= hi)].copy()


def create_covariate_list(
    include_time: bool = False,
    include_humidity: bool = True,
    include_pressure: bool = True,
) -> List[str]:
    """Build standard covariate list for Bangladesh data."""
    covs = ["temp"]
    if include_humidity:
        covs.append("humidity")
    if include_pressure:
        covs.append("surface_pressure")
    if include_time:
        covs.extend(["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"])
    return covs


def standardize_dataframe(
    df: pd.DataFrame,
    target_col: str,
    covariate_cols: List[str],
    train_end_idx: int,
) -> Tuple[pd.DataFrame, object, object]:
    """
    Standardize target and covariates using training data statistics.

    Returns:
        (scaled_df, target_scaler, covariate_scaler)
    """
    from sklearn.preprocessing import StandardScaler
    df = df.copy()
    target_scaler = StandardScaler().fit(df[[target_col]].iloc[:train_end_idx])
    cov_scaler = StandardScaler().fit(df[covariate_cols].iloc[:train_end_idx])

    df[target_col] = target_scaler.transform(df[[target_col]]).ravel()
    df[covariate_cols] = cov_scaler.transform(df[covariate_cols])

    return df, target_scaler, cov_scaler


# Re-export for convenience
__all__ = [
    "load_bangladesh_data",
    "add_time_features",
    "filter_outliers",
    "create_covariate_list",
    "standardize_dataframe",
    "DemandWindowDataset",
    "chronological_split",
]