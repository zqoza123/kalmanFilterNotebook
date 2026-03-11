"""Kalman filter model for dynamic hedge ratio estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KalmanHedgeModel:
    """Two-state Kalman filter for y_t = alpha_t + beta_t * x_t + e_t."""

    process_var: float
    obs_var: float
    init_alpha: float = 0.0
    init_beta: float = 1.0
    initial_cov_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.process_var <= 0:
            raise ValueError("process_var must be > 0")
        if self.obs_var <= 0:
            raise ValueError("obs_var must be > 0")
        if self.initial_cov_scale <= 0:
            raise ValueError("initial_cov_scale must be > 0")
        self.reset()

    def reset(self) -> None:
        self.state = np.array([self.init_alpha, self.init_beta], dtype=float)
        self.cov = np.eye(2, dtype=float) * self.initial_cov_scale

    def update_one(self, y_t: float, x_t: float) -> dict[str, float]:
        """Run one Kalman update step using a single observation."""

        if np.isnan(y_t) or np.isnan(x_t):
            raise ValueError("y_t and x_t must be finite values")

        q = np.eye(2, dtype=float) * self.process_var
        state_pred = self.state.copy()
        cov_pred = self.cov + q

        h = np.array([1.0, float(x_t)], dtype=float)
        innovation = float(y_t - h @ state_pred)
        spread_var = float(h @ cov_pred @ h.T + self.obs_var)
        if spread_var <= 0:
            raise ValueError("Non-positive spread variance encountered.")

        gain = (cov_pred @ h) / spread_var
        state_upd = state_pred + gain * innovation
        cov_upd = (np.eye(2) - np.outer(gain, h)) @ cov_pred

        self.state = state_upd
        self.cov = 0.5 * (cov_upd + cov_upd.T)

        spread = float(y_t - (self.state[0] + self.state[1] * x_t))
        return {
            "alpha": float(self.state[0]),
            "beta": float(self.state[1]),
            "spread": spread,
            "spread_var": spread_var,
            "innovation": innovation,
            "gain_alpha": float(gain[0]),
            "gain_beta": float(gain[1]),
        }

    def fit(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """Fit model over a full aligned time series."""

        frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        if frame.empty:
            raise ValueError("Input series have no overlapping non-null observations.")

        self.reset()
        records = []
        for _, row in frame.iterrows():
            records.append(self.update_one(float(row["y"]), float(row["x"])))

        return pd.DataFrame.from_records(records, index=frame.index)
