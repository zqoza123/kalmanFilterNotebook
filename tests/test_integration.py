from pathlib import Path

import numpy as np
import pandas as pd

from kalman_pairs.backtest import run_backtest
from kalman_pairs.data import fetch_prices
from kalman_pairs.kalman import KalmanHedgeModel
from kalman_pairs.pair_selection import rank_pairs
from kalman_pairs.signals import compute_normalized_innovation, generate_positions


def _create_synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2024-01-02", periods=320)
    factor = np.cumsum(rng.normal(0.0008, 0.012, len(dates)))

    def make_price(base: float, beta: float, noise: float) -> np.ndarray:
        idiosyncratic = np.cumsum(rng.normal(0.0, noise, len(dates)))
        return np.exp(np.log(base) + beta * factor + idiosyncratic)

    return pd.DataFrame(
        {
            "XLF": make_price(35.0, 1.00, 0.003),
            "JPM": make_price(120.0, 1.05, 0.004),
            "BAC": make_price(30.0, 0.99, 0.0038),
            "WFC": make_price(45.0, 1.02, 0.0040),
        },
        index=dates,
    )


def test_offline_end_to_end_pipeline(tmp_path: Path) -> None:
    prices = _create_synthetic_prices()

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_dir / "prices_1d.csv", index_label="Date")

    loaded = fetch_prices(
        tickers=["XLF", "JPM", "BAC", "WFC"],
        start=prices.index.min().strftime("%Y-%m-%d"),
        end=prices.index.max().strftime("%Y-%m-%d"),
        interval="1d",
        source="cache",
        cache_dir=cache_dir,
    )

    ranked = rank_pairs(loaded, top_n=3, min_history=200, max_pvalue=0.05)
    assert not ranked.empty

    top = ranked.iloc[0]
    y = loaded[top["y"]]
    x = loaded[top["x"]]

    model = KalmanHedgeModel(process_var=1e-4, obs_var=1e-2)
    state_df = model.fit(y, x)

    signal = compute_normalized_innovation(state_df["innovation"], state_df["spread_var"], clip=6.0)
    positions = generate_positions(signal, entry=1.2, exit=0.3, stop=3.0, max_hold=15)

    results = run_backtest(
        y=y.loc[state_df.index],
        x=x.loc[state_df.index],
        beta_t=state_df["beta"],
        positions=positions,
        costs_bps=2,
    )

    assert not results["returns"].empty
    assert not results["equity_curve"].empty

    for key in ["Sharpe", "max_drawdown", "hit_rate", "turnover"]:
        assert key in results["metrics"]
        assert np.isfinite(results["metrics"][key])
