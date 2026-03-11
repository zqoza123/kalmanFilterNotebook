"""Data loading and caching helpers for pairs trading."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

try:
    import yfinance as yf
except ModuleNotFoundError:
    class _MissingYFinance:
        def download(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "yfinance is required for source='yfinance'. Install dependencies with "
                "`pip install -e .`."
            )

    yf = _MissingYFinance()


def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
    ordered_unique = []
    seen = set()
    for ticker in tickers:
        value = ticker.strip().upper()
        if not value or value in seen:
            continue
        ordered_unique.append(value)
        seen.add(value)
    if not ordered_unique:
        raise ValueError("At least one ticker is required.")
    return ordered_unique


def _cache_paths(cache_dir: str | Path, interval: str) -> tuple[Path, Path]:
    base = Path(cache_dir)
    return base / f"prices_{interval}.parquet", base / f"prices_{interval}.csv"


def _normalize_close_prices(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("Downloaded dataset is empty.")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        if "Adj Close" in level0:
            close = raw["Adj Close"]
        elif "Close" in level0:
            close = raw["Close"]
        else:
            raise ValueError("Unable to find Close or Adj Close in downloaded data.")
    else:
        if "Close" in raw.columns:
            close = raw[["Close"]]
        elif "Adj Close" in raw.columns:
            close = raw[["Adj Close"]]
        else:
            close = raw.copy()

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    if close.shape[1] == 1 and len(tickers) == 1:
        close.columns = [tickers[0]]

    close.columns = [str(col).upper() for col in close.columns]
    close = close.sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def _slice_prices(
    prices: pd.DataFrame,
    tickers: Sequence[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError("start must be earlier than or equal to end.")

    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise KeyError(f"Cache is missing tickers: {missing}")

    sliced = prices.loc[(prices.index >= start_ts) & (prices.index <= end_ts), list(tickers)]
    if sliced.empty:
        raise ValueError("No rows available for requested date range.")
    return sliced


def _write_cache(prices: pd.DataFrame, cache_dir: str | Path, interval: str) -> None:
    parquet_path, csv_path = _cache_paths(cache_dir, interval)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    prices.sort_index().to_csv(csv_path, index_label="Date")
    try:
        prices.sort_index().to_parquet(parquet_path)
    except Exception:
        # CSV cache remains the portable fallback when parquet engine is missing.
        pass


def load_cached_prices(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    interval: str,
    cache_dir: str | Path = "data/cache",
) -> pd.DataFrame:
    """Load prices from local cache (parquet preferred, CSV fallback)."""

    normalized_tickers = _normalize_tickers(tickers)
    parquet_path, csv_path = _cache_paths(cache_dir, interval)

    if parquet_path.exists():
        try:
            prices = pd.read_parquet(parquet_path)
            return _slice_prices(prices, normalized_tickers, start, end)
        except Exception:
            pass

    if csv_path.exists():
        prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        prices.columns = [str(col).upper() for col in prices.columns]
        return _slice_prices(prices, normalized_tickers, start, end)

    raise FileNotFoundError(
        f"No cache file found in {Path(cache_dir).resolve()} for interval={interval}."
    )


def fetch_prices(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    interval: str = "1d",
    source: str = "yfinance",
    force_refresh: bool = False,
    cache_dir: str | Path = "data/cache",
) -> pd.DataFrame:
    """Fetch prices from yfinance or local cache with cache-first behavior."""

    normalized_tickers = _normalize_tickers(tickers)
    source_key = source.lower()

    if source_key == "cache":
        return load_cached_prices(normalized_tickers, start, end, interval, cache_dir)

    if source_key != "yfinance":
        raise ValueError("source must be either 'yfinance' or 'cache'.")

    if not force_refresh:
        try:
            return load_cached_prices(normalized_tickers, start, end, interval, cache_dir)
        except (FileNotFoundError, KeyError, ValueError):
            pass

    raw = yf.download(
        tickers=normalized_tickers,
        start=str(pd.Timestamp(start).date()),
        end=str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()),
        interval=interval,
        progress=False,
        auto_adjust=True,
        group_by="column",
        threads=True,
    )
    close = _normalize_close_prices(raw, normalized_tickers)

    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        historical = load_cached_prices(
            normalized_tickers,
            start="1900-01-01",
            end="2100-01-01",
            interval=interval,
            cache_dir=cache_dir,
        )
        outside_requested_window = historical.loc[
            (historical.index < start_ts) | (historical.index > end_ts)
        ]
        close = pd.concat([outside_requested_window, close]).sort_index()
        close = close[~close.index.duplicated(keep="last")]
    except (FileNotFoundError, KeyError, ValueError):
        pass

    close = close.ffill().dropna(how="all")
    if close.empty:
        raise ValueError("No close prices available after preprocessing.")

    _write_cache(close, cache_dir, interval)
    return _slice_prices(close, normalized_tickers, start, end)
