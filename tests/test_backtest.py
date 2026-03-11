import pandas as pd

from kalman_pairs.backtest import run_backtest


def test_backtest_costs_reduce_total_return() -> None:
    index = pd.bdate_range("2025-01-01", periods=8)

    y = pd.Series([100, 101, 102, 101, 103, 104, 103, 105], index=index)
    x = pd.Series([50, 50.5, 51, 50.8, 51.5, 52, 51.7, 52.4], index=index)
    beta = pd.Series([2.0] * len(index), index=index)
    positions = pd.Series([0, 1, 1, 0, -1, -1, 0, 0], index=index)

    no_cost = run_backtest(y=y, x=x, beta_t=beta, positions=positions, costs_bps=0)
    with_cost = run_backtest(y=y, x=x, beta_t=beta, positions=positions, costs_bps=5)

    assert with_cost["returns"].sum() < no_cost["returns"].sum()
    assert not with_cost["trades"].empty

    for key in ["Sharpe", "max_drawdown", "hit_rate", "turnover"]:
        assert key in with_cost["metrics"]
