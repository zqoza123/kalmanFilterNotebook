import numpy as np
import pandas as pd

from kalman_pairs.kalman import KalmanHedgeModel


def test_kalman_fit_recovers_stable_beta() -> None:
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2024-01-01", periods=280)

    x = pd.Series(np.linspace(10.0, 60.0, len(index)), index=index, name="x")
    noise = rng.normal(0.0, 0.35, len(index))
    y = pd.Series(1.5 + 2.0 * x.values + noise, index=index, name="y")

    model = KalmanHedgeModel(process_var=1e-4, obs_var=0.2)
    fitted = model.fit(y, x)

    assert abs(float(fitted["beta"].iloc[-1]) - 2.0) < 0.15


def test_update_one_returns_expected_payload() -> None:
    model = KalmanHedgeModel(process_var=1e-4, obs_var=0.1)
    update = model.update_one(y_t=101.0, x_t=50.0)

    for key in [
        "alpha",
        "beta",
        "spread",
        "spread_var",
        "innovation",
        "gain_alpha",
        "gain_beta",
    ]:
        assert key in update
