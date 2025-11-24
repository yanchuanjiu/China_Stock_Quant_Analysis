"""Data pipeline to pull TongHuaShun data via akshare and convert to Qlib format."""
from __future__ import annotations

import argparse
import importlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, TYPE_CHECKING

from .config import (
    AKSHARE_DATA_DIR,
    DEFAULT_DATE_RANGE,
    DEFAULT_FREQ,
    DEFAULT_YAHOO_PROVIDER,
    QLIB_PROVIDER_URI,
    ensure_directories,
)

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    import pandas as pd


class MissingDependencyError(RuntimeError):
    """Raised when a required third-party dependency is missing."""


def require_dependencies():
    """Import heavy, network-installed dependencies lazily.

    This keeps `python -m qlib_cn_lowfreq.data_pipeline --help` usable even when
    akshare/qlib/pandas are not preinstalled (common in sandboxed CI). A helpful
    error is raised the moment the pipeline actually needs them.
    """

    try:
        ak = importlib.import_module("akshare")
        pd = importlib.import_module("pandas")
        qlib = importlib.import_module("qlib")
        dataset_utils = importlib.import_module("qlib.data.dataset.utils")
        workflow_mod = importlib.import_module("qlib.workflow")
    except ImportError as exc:  # pragma: no cover - exercised in environments without deps
        raise MissingDependencyError(
            "缺少依赖：请先安装 akshare、pandas、pyqlib 等数据处理依赖。"
            " 如果网络受限，可预先下载离线 wheel 并使用 `pip install --no-index --find-links <dir> -r requirements.txt`。"
        ) from exc

    return {
        "ak": ak,
        "pd": pd,
        "qlib": qlib,
        "convert_index_format": dataset_utils.convert_index_format,
        "dump_bin": dataset_utils.dump_bin,
        "R": workflow_mod.R,
    }


@dataclass
class TongHuaShunFetcher:
    """Fetches historical A-share data from akshare's TongHuaShun interface."""

    start: str
    end: str
    symbols: Optional[List[str]] = None
    adjust: str = "qfq"

    def __post_init__(self) -> None:
        deps = require_dependencies()
        self.ak = deps["ak"]
        self.pd = deps["pd"]

    def list_instruments(self) -> List[str]:
        """List instruments using akshare if not provided."""

        if self.symbols:
            return self.symbols
        listing = self.ak.stock_zh_a_spot_em()
        return [
            f"{row['代码']}.SH" if row['代码'].startswith("6") else f"{row['代码']}.SZ"
            for _, row in listing.iterrows()
        ]

    def fetch_symbol(self, symbol: str):
        """Fetch a single symbol's daily history and format columns."""

        pd = self.pd
        ts_code = symbol.replace(".SH", "").replace(".SZ", "")
        df = self.ak.stock_zh_a_hist_ths(
            symbol=ts_code,
            start_date=self.start,
            end_date=self.end,
            adjust=self.adjust,
        )
        if df.empty:
            raise ValueError(f"No data returned for {symbol} between {self.start} and {self.end}")

        df.rename(
            columns={
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
            },
            inplace=True,
        )

        df["date"] = pd.to_datetime(df["日期"])
        df.set_index("date", inplace=True)
        df = df[["open", "high", "low", "close", "volume", "amount"]]
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        df["symbol"] = symbol
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df.dropna(subset=numeric_cols, inplace=True)
        return df

    def fetch_all(self):
        """Fetch and concatenate all instruments."""

        pd = self.pd
        frames = [self.fetch_symbol(symbol) for symbol in self.list_instruments()]
        return pd.concat(frames).sort_index()


def dump_to_qlib_store(df, provider_uri: Path, mirror_to: Optional[Path] = None) -> None:
    """Dump dataframe into Qlib binary store using the standard data handler.

    This uses Qlib's `dump_bin` utility so that the generated目录可以直接与官方
    Yahoo 财经日频数据合并。默认写入日频 (`day`) 二进制格式。
    """

    if df.empty:
        raise ValueError("Input dataframe is empty; nothing to dump")

    provider_uri.mkdir(parents=True, exist_ok=True)
    deps = require_dependencies()
    qlib = deps["qlib"]
    convert_index_format = deps["convert_index_format"]
    dump_bin = deps["dump_bin"]

    qlib.init(provider_uri=str(provider_uri), region="cn")

    prepared = (
        df.copy()
        .reset_index()
        .rename(columns={"date": "datetime"})
        .set_index(["symbol", "datetime"])
    )
    prepared.index.names = ["instrument", "datetime"]
    prepared = prepared.sort_index()
    prepared = convert_index_format(prepared)

    dump_bin(prepared, root_dir=str(provider_uri), freq=DEFAULT_FREQ, exclude_fields=None)

    if mirror_to is not None:
        mirror_to = Path(mirror_to)
        mirror_to.mkdir(parents=True, exist_ok=True)
        shutil.copytree(provider_uri, mirror_to, dirs_exist_ok=True)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Akshare TongHuaShun to Qlib pipeline")
    parser.add_argument("--start", default=DEFAULT_DATE_RANGE.start, help="Start date, e.g., 2015-01-01")
    parser.add_argument("--end", default=DEFAULT_DATE_RANGE.end, help="End date, e.g., 2024-12-31")
    parser.add_argument("--symbols", nargs="*", help="Optional list of symbols like 000001.SZ 600000.SH")
    parser.add_argument(
        "--provider-uri",
        default=str(QLIB_PROVIDER_URI),
        help="Where to store the generated Qlib data (default: data/akshare_ths/qlib_data)",
    )
    parser.add_argument(
        "--mirror-yahoo",
        action="store_true",
        help="Also mirror the生成结果到默认的 Yahoo 财经 Qlib 数据目录，便于联合使用",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    ensure_directories()

    deps = require_dependencies()
    R = deps["R"]

    fetcher = TongHuaShunFetcher(start=args.start, end=args.end, symbols=args.symbols)
    raw_df = fetcher.fetch_all()

    processed_df = raw_df
    processed_df.sort_index(inplace=True)

    mirror_target = DEFAULT_YAHOO_PROVIDER if args.mirror_yahoo else None
    dump_to_qlib_store(processed_df, provider_uri=Path(args.provider_uri), mirror_to=mirror_target)

    mlruns_uri = (AKSHARE_DATA_DIR / "mlruns").absolute()
    with R.start(exp_name="data_pipeline", recorder_name="dump", uri=str(mlruns_uri)) as recorder:
        recorder.log_params({
            "start": args.start,
            "end": args.end,
            "symbols": args.symbols,
            "mirror_yahoo": args.mirror_yahoo,
            "provider_uri": args.provider_uri,
        })
        recorder.log_artifact(str(Path(args.provider_uri).absolute()))


if __name__ == "__main__":
    main()
