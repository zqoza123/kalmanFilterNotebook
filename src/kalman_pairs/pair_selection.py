"""Pair ranking utilities."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def _pair_statistics(y: pd.Series, x: pd.Series) -> dict[str, float | str]:
    frame = pd.concat([y, x], axis=1).dropna()
    frame.columns = ["y", "x"]

    y_values = frame["y"]
    x_values = frame["x"]

    _, pvalue, _ = coint(y_values, x_values)

    model = sm.OLS(y_values, sm.add_constant(x_values)).fit()
    alpha = float(model.params.iloc[0])
    beta = float(model.params.iloc[1])

    spread = y_values - (alpha + beta * x_values)
    spread_std = float(spread.std(ddof=0))
    spread_autocorr = float(spread.autocorr(lag=1)) if len(spread) > 2 else 0.0
    if np.isnan(spread_autocorr):
        spread_autocorr = 0.0

    mean_reversion_speed = max(0.0, 1.0 - abs(spread_autocorr))
    score = (1.0 - min(max(float(pvalue), 0.0), 1.0)) * 0.7 + mean_reversion_speed * 0.3

    return {
        "y": str(y.name),
        "x": str(x.name),
        "n_obs": int(len(frame)),
        "pvalue": float(pvalue),
        "alpha": alpha,
        "beta": beta,
        "spread_std": spread_std,
        "spread_autocorr": spread_autocorr,
        "mean_reversion_speed": mean_reversion_speed,
        "score": score,
    }


def rank_pairs(
    prices: pd.DataFrame,
    top_n: int = 5,
    min_history: int = 252,
    max_pvalue: float = 1.0,
) -> pd.DataFrame:
    """Rank pairs by cointegration and spread stability diagnostics."""

    if top_n <= 0:
        raise ValueError("top_n must be > 0")
    if min_history < 30:
        raise ValueError("min_history must be >= 30")
    if not (0.0 < max_pvalue <= 1.0):
        raise ValueError("max_pvalue must be in (0, 1].")

    clean = prices.sort_index().copy()
    if clean.shape[1] < 2:
        raise ValueError("At least two tickers are required to rank pairs.")

    records: list[dict[str, float | str]] = []
    for y_col, x_col in combinations(clean.columns, 2):
        pair_frame = clean[[y_col, x_col]].dropna()
        if len(pair_frame) < min_history:
            continue

        try:
            stats = _pair_statistics(pair_frame[y_col], pair_frame[x_col])
        except Exception:
            continue
        records.append(stats)

    columns = [
        "y",
        "x",
        "n_obs",
        "pvalue",
        "alpha",
        "beta",
        "spread_std",
        "spread_autocorr",
        "mean_reversion_speed",
        "score",
    ]

    if not records:
        return pd.DataFrame(columns=columns)

    ranked = pd.DataFrame.from_records(records)
    ranked = ranked[ranked["pvalue"] <= max_pvalue].copy()
    if ranked.empty:
        return pd.DataFrame(columns=columns)
    ranked = ranked.sort_values(["score", "pvalue"], ascending=[False, True]).reset_index(drop=True)
    return ranked.head(top_n)
