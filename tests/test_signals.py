import pandas as pd

from kalman_pairs.signals import (
    compute_normalized_innovation,
    compute_spread_zscore,
    generate_positions,
)


def test_compute_spread_zscore_returns_expected_length() -> None:
    spread = pd.Series([0.1, 0.2, -0.1, 0.0, 0.3, -0.2, 0.1])
    z = compute_spread_zscore(spread, lookback=5)

    assert len(z) == len(spread)
    assert z.iloc[:4].eq(0.0).all()


def test_generate_positions_entry_exit_stop_rules() -> None:
    z = pd.Series([0.0, -2.2, -1.8, -0.3, 0.0, 2.1, 3.8, 0.0])
    positions = generate_positions(z, entry=2.0, exit=0.5, stop=3.5, max_hold=5)

    assert positions.tolist() == [0, 1, 1, 0, 0, -1, 0, 0]


def test_generate_positions_respects_max_hold() -> None:
    z = pd.Series([-2.3, -2.1, -2.2, -2.0, -2.4])
    positions = generate_positions(z, entry=2.0, exit=0.5, stop=3.5, max_hold=2)

    assert positions.tolist() == [1, 1, 0, 1, 1]


def test_compute_normalized_innovation_matches_expected_values() -> None:
    innovation = pd.Series([0.0, 2.0, -3.0], index=pd.RangeIndex(3))
    spread_var = pd.Series([1.0, 4.0, 9.0], index=pd.RangeIndex(3))

    signal = compute_normalized_innovation(innovation, spread_var)

    assert signal.tolist() == [0.0, 1.0, -1.0]


def test_compute_normalized_innovation_handles_zero_variance() -> None:
    innovation = pd.Series([1.0, -1.0], index=pd.RangeIndex(2))
    spread_var = pd.Series([0.0, 0.0], index=pd.RangeIndex(2))

    signal = compute_normalized_innovation(innovation, spread_var)

    assert signal.tolist() == [0.0, 0.0]
