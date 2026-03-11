import pandas as pd

import kalman_pairs.pair_selection as pair_selection


def test_rank_pairs_applies_max_pvalue_filter(monkeypatch) -> None:
    dates = pd.bdate_range("2025-01-01", periods=60)
    prices = pd.DataFrame(
        {
            "AAA": range(60),
            "BBB": range(1, 61),
            "CCC": range(2, 62),
        },
        index=dates,
    )

    lookup = {
        ("AAA", "BBB"): 0.01,
        ("AAA", "CCC"): 0.10,
        ("BBB", "CCC"): 0.03,
    }

    def fake_pair_statistics(y: pd.Series, x: pd.Series):
        pvalue = lookup[(y.name, x.name)]
        score = 1.0 - pvalue
        return {
            "y": y.name,
            "x": x.name,
            "n_obs": len(y),
            "pvalue": pvalue,
            "alpha": 0.0,
            "beta": 1.0,
            "spread_std": 1.0,
            "spread_autocorr": 0.0,
            "mean_reversion_speed": 1.0,
            "score": score,
        }

    monkeypatch.setattr(pair_selection, "_pair_statistics", fake_pair_statistics)

    ranked = pair_selection.rank_pairs(prices, top_n=5, min_history=30, max_pvalue=0.05)

    assert not ranked.empty
    assert (ranked["pvalue"] <= 0.05).all()
    assert set(zip(ranked["y"], ranked["x"])) == {("AAA", "BBB"), ("BBB", "CCC")}


def test_rank_pairs_rejects_invalid_max_pvalue() -> None:
    prices = pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]})

    try:
        pair_selection.rank_pairs(prices, max_pvalue=0.0)
    except ValueError as exc:
        assert "max_pvalue" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid max_pvalue")
