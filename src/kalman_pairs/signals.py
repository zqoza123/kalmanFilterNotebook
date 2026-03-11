"""Signal generation from filtered spreads."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spread_zscore(spread: pd.Series, lookback: int = 60) -> pd.Series:
    """Rolling z-score of spread."""

    if lookback < 5:
        raise ValueError("lookback must be >= 5")

    rolling_mean = spread.rolling(lookback, min_periods=lookback).mean()
    rolling_std = spread.rolling(lookback, min_periods=lookback).std(ddof=0)
    zscore = (spread - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return zscore.fillna(0.0)


def compute_normalized_innovation(
    innovation: pd.Series,
    spread_var: pd.Series,
    clip: float | None = None,
) -> pd.Series:
    """Innovation normalized by forecast spread volatility."""

    if not innovation.index.equals(spread_var.index):
        spread_var = spread_var.reindex(innovation.index)

    denom = np.sqrt(spread_var.astype(float).clip(lower=0.0))
    signal = innovation.astype(float) / denom.replace(0.0, np.nan)
    signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if clip is not None:
        if clip <= 0:
            raise ValueError("clip must be > 0 when provided.")
        signal = signal.clip(lower=-clip, upper=clip)

    return signal.rename("normalized_innovation")


def generate_positions(
    z: pd.Series,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 3.5,
    max_hold: int = 20,
) -> pd.Series:
    """Stateful position engine for mean-reversion trading."""

    if not (entry > exit > 0):
        raise ValueError("Thresholds must satisfy entry > exit > 0.")
    if stop <= entry:
        raise ValueError("stop must be greater than entry.")
    if max_hold <= 0:
        raise ValueError("max_hold must be > 0.")

    positions: list[int] = []
    current = 0
    holding_days = 0

    for value in z.fillna(0.0):
        value = float(value)
        abs_value = abs(value)

        if current == 0:
            if value <= -entry:
                current = 1
                holding_days = 1
            elif value >= entry:
                current = -1
                holding_days = 1
        else:
            holding_days += 1
            if abs_value <= exit or abs_value >= stop or holding_days > max_hold:
                current = 0
                holding_days = 0

        positions.append(current)

    return pd.Series(positions, index=z.index, name="position", dtype="int8")
