"""Kalman pairs trading toolkit."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "KalmanHedgeModel",
    "compute_normalized_innovation",
    "compute_spread_zscore",
    "fetch_prices",
    "generate_positions",
    "load_cached_prices",
    "rank_pairs",
    "run_backtest",
]

_EXPORT_MAP = {
    "KalmanHedgeModel": ("kalman_pairs.kalman", "KalmanHedgeModel"),
    "compute_normalized_innovation": ("kalman_pairs.signals", "compute_normalized_innovation"),
    "compute_spread_zscore": ("kalman_pairs.signals", "compute_spread_zscore"),
    "fetch_prices": ("kalman_pairs.data", "fetch_prices"),
    "generate_positions": ("kalman_pairs.signals", "generate_positions"),
    "load_cached_prices": ("kalman_pairs.data", "load_cached_prices"),
    "rank_pairs": ("kalman_pairs.pair_selection", "rank_pairs"),
    "run_backtest": ("kalman_pairs.backtest", "run_backtest"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
