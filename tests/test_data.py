from pathlib import Path

import numpy as np
import pandas as pd

import kalman_pairs.data as data_module


def _write_cache_file(cache_dir: Path, prices: pd.DataFrame, interval: str = "1d") -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"prices_{interval}.csv"
    prices.to_csv(path, index_label="Date")
    return path


def test_load_cached_prices_returns_filtered_subset(tmp_path: Path) -> None:
    dates = pd.bdate_range("2025-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "AAPL": [100, 101, 102, 103, 104],
            "MSFT": [200, 201, 202, 203, 204],
        },
        index=dates,
    )
    _write_cache_file(tmp_path / "cache", prices)

    out = data_module.load_cached_prices(
        tickers=["AAPL", "MSFT"],
        start="2025-01-02",
        end="2025-01-06",
        interval="1d",
        cache_dir=tmp_path / "cache",
    )

    assert list(out.columns) == ["AAPL", "MSFT"]
    assert len(out) == 3


def test_fetch_prices_uses_cache_without_network(tmp_path: Path, monkeypatch) -> None:
    dates = pd.bdate_range("2025-01-01", periods=6)
    prices = pd.DataFrame(
        {
            "AAPL": np.linspace(100, 105, len(dates)),
            "MSFT": np.linspace(200, 210, len(dates)),
        },
        index=dates,
    )
    cache_dir = tmp_path / "cache"
    _write_cache_file(cache_dir, prices)

    calls = {"count": 0}

    def fake_download(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("yfinance.download should not be called when cache satisfies request")

    monkeypatch.setattr(data_module.yf, "download", fake_download)

    out = data_module.fetch_prices(
        tickers=["AAPL", "MSFT"],
        start="2025-01-01",
        end="2025-01-08",
        interval="1d",
        source="yfinance",
        force_refresh=False,
        cache_dir=cache_dir,
    )

    assert calls["count"] == 0
    assert not out.empty


def test_fetch_prices_force_refresh_downloads_and_writes_cache(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"

    dates = pd.bdate_range("2025-02-03", periods=5)
    close = pd.DataFrame(
        np.column_stack(
            [np.linspace(150.0, 155.0, len(dates)), np.linspace(300.0, 304.0, len(dates))]
        ),
        index=dates,
        columns=pd.MultiIndex.from_product([["Close"], ["AAPL", "MSFT"]]),
    )

    calls = {"count": 0}

    def fake_download(*args, **kwargs):
        calls["count"] += 1
        return close

    monkeypatch.setattr(data_module.yf, "download", fake_download)

    out = data_module.fetch_prices(
        tickers=["AAPL", "MSFT"],
        start="2025-02-03",
        end="2025-02-10",
        interval="1d",
        source="yfinance",
        force_refresh=True,
        cache_dir=cache_dir,
    )

    assert calls["count"] == 1
    assert not out.empty
    assert (cache_dir / "prices_1d.csv").exists()


def test_fetch_prices_force_refresh_replaces_cached_rows_in_requested_window(
    tmp_path: Path, monkeypatch
) -> None:
    cache_dir = tmp_path / "cache"
    cached_dates = pd.bdate_range("2025-03-03", periods=5)
    cached = pd.DataFrame(
        {
            "AAPL": np.linspace(100.0, 104.0, len(cached_dates)),
            "MSFT": np.linspace(200.0, 204.0, len(cached_dates)),
        },
        index=cached_dates,
    )
    _write_cache_file(cache_dir, cached)

    live_dates = pd.DatetimeIndex(["2025-03-04", "2025-03-06"])
    live = pd.DataFrame(
        np.column_stack(
            [np.array([150.0, 151.0]), np.array([300.0, 301.0])]
        ),
        index=live_dates,
        columns=pd.MultiIndex.from_product([["Close"], ["AAPL", "MSFT"]]),
    )

    def fake_download(*args, **kwargs):
        return live

    monkeypatch.setattr(data_module.yf, "download", fake_download)

    out = data_module.fetch_prices(
        tickers=["AAPL", "MSFT"],
        start="2025-03-03",
        end="2025-03-07",
        interval="1d",
        source="yfinance",
        force_refresh=True,
        cache_dir=cache_dir,
    )

    assert list(out.index) == list(live_dates)
    assert float(out.loc[pd.Timestamp("2025-03-04"), "AAPL"]) == 150.0
