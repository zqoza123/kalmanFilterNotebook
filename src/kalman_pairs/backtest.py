"""Simple daily backtester for pairs strategy outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    y: pd.Series,
    x: pd.Series,
    beta_t: pd.Series,
    positions: pd.Series,
    costs_bps: float = 2.0,
) -> dict[str, object]:
    """Run backtest and return returns, trades, metrics, and equity curve."""

    if costs_bps < 0:
        raise ValueError("costs_bps must be non-negative.")

    frame = pd.concat(
        [
            y.rename("y"),
            x.rename("x"),
            beta_t.rename("beta"),
            positions.rename("position"),
        ],
        axis=1,
    ).dropna()

    if frame.empty:
        raise ValueError("Backtest inputs have no overlapping rows.")

    frame["ret_y"] = frame["y"].pct_change().fillna(0.0)
    frame["ret_x"] = frame["x"].pct_change().fillna(0.0)
    frame["beta_lag"] = frame["beta"].shift(1).ffill().fillna(frame["beta"])

    frame["pair_ret"] = frame["ret_y"] - frame["beta_lag"] * frame["ret_x"]
    frame["position_lag"] = frame["position"].shift(1).fillna(0.0)

    frame["gross_ret"] = frame["position_lag"] * frame["pair_ret"]
    frame["trade_size"] = frame["position"].diff().abs().fillna(frame["position"].abs())

    cost_rate = costs_bps / 10_000.0
    frame["cost"] = frame["trade_size"] * cost_rate
    frame["net_ret"] = frame["gross_ret"] - frame["cost"]

    frame["equity_curve"] = (1.0 + frame["net_ret"]).cumprod()
    frame["drawdown"] = frame["equity_curve"] / frame["equity_curve"].cummax() - 1.0

    returns = frame["net_ret"]
    vol = float(returns.std(ddof=0))
    sharpe = float(np.sqrt(252.0) * returns.mean() / vol) if vol > 0 else 0.0

    active = frame.loc[frame["position_lag"] != 0, "net_ret"]
    hit_rate = float((active > 0).mean()) if not active.empty else 0.0

    metrics = {
        "Sharpe": sharpe,
        "max_drawdown": float(frame["drawdown"].min()),
        "hit_rate": hit_rate,
        "turnover": float(frame["trade_size"].sum() / len(frame)),
        "total_return": float(frame["equity_curve"].iloc[-1] - 1.0),
        "trades": int((frame["trade_size"] > 0).sum()),
    }

    trades = frame.loc[
        frame["trade_size"] > 0,
        ["position", "trade_size", "cost", "gross_ret", "net_ret"],
    ].copy()

    equity_curve = frame[["equity_curve", "drawdown"]].copy()

    return {
        "returns": returns,
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
    }
