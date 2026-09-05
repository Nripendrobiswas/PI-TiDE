"""
Physics Constraints for Load Forecasting
========================================

Implements physical constraints for electricity demand forecasting:
- Heating/Cooling Degree-Day (HDD/CDD) sensitivity regimes
- Temperature-dependent capacity envelope (P_max(T))
- Ramp rate limits
- Non-negativity
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Tuple
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression


class DegreeDayPhysics:
    """
    Encodes physical priors for temperature-demand relationship.

    For grids with distinct heating/cooling regimes:
    - Below balance temperature T_b: demand rises as temperature falls (heating)
    - Above T_b: demand rises as temperature rises (cooling)

    For monotonic (cooling-only) grids like Bangladesh:
    - cooling_only=True forces positive sensitivity everywhere
    """

    def __init__(self, balance_temp: float = 22.0, cooling_only: bool = False):
        self.T_b = balance_temp
        self.cooling_only = cooling_only
        # Temperature-power envelope (fitted from data)
        self.envelope_temps: Optional[np.ndarray] = None
        self.envelope_max_power: Optional[np.ndarray] = None

    def hdd(self, temp: torch.Tensor) -> torch.Tensor:
        """Heating Degree Days: max(T_b - T, 0)"""
        return torch.clamp(self.T_b - temp, min=0.0)

    def cdd(self, temp: torch.Tensor) -> torch.Tensor:
        """Cooling Degree Days: max(T - T_b, 0)"""
        return torch.clamp(temp - self.T_b, min=0.0)

    def expected_sensitivity_sign(self, temp: torch.Tensor) -> torch.Tensor:
        """
        Expected sign of d(demand)/d(temp):
        - -1 in heating regime (T < T_b)
        - +1 in cooling regime (T >= T_b)
        - cooling_only=True forces +1 everywhere
        """
        if self.cooling_only:
            return torch.ones_like(temp)
        return torch.where(temp < self.T_b, -torch.ones_like(temp), torch.ones_like(temp))

    def fit_envelope(
        self,
        df: pd.DataFrame,
        temp_col: str = "temp",
        target_col: str = "demand",
        n_bins: int = 30,
    ) -> None:
        """
        Fit piecewise-linear upper envelope P_max(T) from binned maximum demand.

        Args:
            df: DataFrame with temperature and demand columns
            temp_col: Temperature column name
            target_col: Demand column name
            n_bins: Number of temperature bins
        """
        lo, hi = df[temp_col].min(), df[temp_col].max()
        bins = np.linspace(lo, hi, n_bins + 1)
        df = df.copy()
        df["tbin"] = pd.cut(df[temp_col], bins=bins)
        max_power = df.groupby("tbin", observed=True)[target_col].max()
        # Interpolate missing bins
        max_power = max_power.interpolate().bfill().ffill()
        self.envelope_temps = np.array([b.mid for b in max_power.index], dtype=np.float32)
        self.envelope_max_power = max_power.values.astype(np.float32)

    def max_power_at_temp(self, temp: torch.Tensor) -> torch.Tensor:
        """
        Interpolate capacity envelope at given temperatures.

        Args:
            temp: Temperature tensor (any shape)
        Returns:
            Maximum power tensor (same shape)
        """
        if self.envelope_temps is None:
            return torch.full_like(temp, float("inf"))
        temp_np = temp.detach().cpu().numpy().ravel()
        interp = np.interp(temp_np, self.envelope_temps, self.envelope_max_power)
        return torch.from_numpy(interp).to(temp.device).reshape(temp.shape)


def estimate_balance_temp_piecewise(
    df: pd.DataFrame,
    target_col: str,
    temp_col: str,
) -> float:
    """
    Estimate balance temperature via segmented regression.

    Fits: demand = a1*T + b1 (T < T_b) + a2*T + b2 (T >= T_b)
    Returns T_b minimizing total RSS.

    Args:
        df: DataFrame with temperature and demand
        target_col: Demand column name
        temp_col: Temperature column name

    Returns:
        Optimal balance temperature (float)
    """
    def rss(tb: float) -> float:
        cold = df[df[temp_col] < tb]
        hot = df[df[temp_col] >= tb]
        if len(cold) < 10 or len(hot) < 10:
            return np.inf
        m1 = LinearRegression().fit(cold[[temp_col]], cold[target_col])
        m2 = LinearRegression().fit(hot[[temp_col]], hot[target_col])
        rss1 = np.sum((cold[target_col] - m1.predict(cold[[temp_col]])) ** 2)
        rss2 = np.sum((hot[target_col] - m2.predict(hot[[temp_col]])) ** 2)
        return rss1 + rss2

    lo, hi = df[temp_col].quantile(0.1), df[temp_col].quantile(0.9)
    res = minimize_scalar(rss, bounds=(lo, hi), method='bounded')
    return float(res.x)