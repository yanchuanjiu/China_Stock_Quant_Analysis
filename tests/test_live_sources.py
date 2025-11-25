"""Live data-source checks using real akshare and Yahoo! Finance data.

These tests are skipped by default to avoid network access in CI. Set the
environment variable ``LIVE_DATA_TESTS=1`` to enable them when validating data
formats end-to-end.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("LIVE_DATA_TESTS") != "1",
    reason="Set LIVE_DATA_TESTS=1 to run live data validations",
)


def _skip_on_network_error(exc: Exception) -> None:
    """Skip gracefully when the host cannot reach public data sources."""

    message = str(exc).lower()
    network_indicators = ["proxy", "timeout", "timed out", "connection", "network", "tls"]
    if isinstance(exc, (OSError, ConnectionError, TimeoutError)) or any(token in message for token in network_indicators):
        pytest.skip(f"Network unavailable for live data validation: {exc}")


def test_akshare_updates_byd_data():
    """Pull real BYD (002594.SZ) data via akshare and validate formatting."""

    pd = pytest.importorskip("pandas")
    pytest.importorskip("akshare")

    from qlib_cn_lowfreq.data_pipeline import TongHuaShunFetcher

    fetcher = TongHuaShunFetcher(start="2024-01-01", end="2024-03-01", symbols=["002594.SZ"])
    try:
        df = fetcher.fetch_all()
    except Exception as exc:  # pragma: no cover - exercised only when LIVE_DATA_TESTS=1
        _skip_on_network_error(exc)
        raise

    assert not df.empty, "Akshare should return live BYD data"
    assert df.index.is_monotonic_increasing
    assert set(["open", "high", "low", "close", "volume", "amount", "symbol"]).issubset(df.columns)
    assert (df["symbol"] == "002594.SZ").all()
    assert pd.api.types.is_datetime64_any_dtype(df.index)
    assert df[["open", "high", "low", "close"]].apply(pd.api.types.is_numeric_dtype).all()


def test_qlib_yahoo_loader_reads_byd(tmp_path: Path):
    """Download Yahoo! Finance dataset via qlib helpers and read BYD quotes."""

    pd = pytest.importorskip("pandas")
    qlib = pytest.importorskip("qlib")
    data_mod = pytest.importorskip("qlib.tests.data")
    from qlib.data import D

    provider = tmp_path / "qlib_data"
    GetData = data_mod.GetData
    try:
        GetData(delete_zip_file=True).qlib_data(
            name="qlib_data_simple", target_dir=provider, region="cn", interval="1d", delete_old=False
        )
    except Exception as exc:  # pragma: no cover - exercised only when LIVE_DATA_TESTS=1
        _skip_on_network_error(exc)
        raise

    qlib.init(provider_uri=str(provider.resolve()), region="cn")
    df = D.features(["002594.SZ"], ["$close", "$volume"], start_time="2023-01-01", end_time="2023-03-01")

    assert not df.empty, "Qlib Yahoo loader should deliver real BYD quotes"
    assert {"$close", "$volume"}.issubset(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df.index)
