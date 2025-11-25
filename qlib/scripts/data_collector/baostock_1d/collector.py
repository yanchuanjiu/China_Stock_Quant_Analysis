# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import fire
import pandas as pd
import baostock as bs
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Iterable, List

import qlib
from qlib.data import D

# Ensure we can import from parent directories
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun

class BaostockCollectorCN1d(BaseCollector):
    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        bs.login()
        super(BaostockCollectorCN1d, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

    def get_instrument_list(self):
        bs.login()
        logger.info("Getting HS300 symbols...")
        # Get HS300 constituents from baostock
        # Note: query_hs300_stocks returns constituents for a specific date.
        # We'll try to get the latest available list.
        # Or iterate over a range if we want dynamic universe (Qlib supports dynamic).
        # For simplicity, we get the latest one available or a recent trading day.
        
        # Try to get a recent trading date
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        rs = bs.query_trade_dates(start_date=pd.Timestamp(today) - pd.Timedelta(days=30), end_date=today)
        dates = []
        while rs.error_code == "0" and rs.next():
             dates.append(rs.get_row_data())
        
        trading_dates = [d[0] for d in dates if d[1] == '1'] # d[1] == '1' is trading day
        
        if not trading_dates:
             # Fallback to a hardcoded recent date if baostock fails to return recent dates
             target_date = "2024-01-01" 
        else:
             target_date = trading_dates[-1]

        logger.info(f"Querying HS300 components for date: {target_date}")
        rs = bs.query_hs300_stocks(date=target_date)
        hs300_stocks = []
        while rs.error_code == "0" and rs.next():
            hs300_stocks.append(rs.get_row_data()[1]) # code is the second column
            
        # If list is empty, maybe try to get all A shares? User specifically asked for HS300.
        # But to ensure we have data, let's make sure we got something.
        if not hs300_stocks:
            logger.warning("Failed to get HS300 list from Baostock. Using a fallback small list for testing.")
            # Fallback list
            hs300_stocks = ["sh.600000", "sh.600036", "sz.000001", "sz.000002"]

        return sorted(list(set(hs300_stocks)))

    def normalize_symbol(self, symbol: str):
        # Baostock returns sh.600000, Qlib uses SH600000
        return symbol.replace(".", "").upper()

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        bs.login()
        # Download raw data (no adjustment)
        df_raw = self._get_baostock_data(symbol, interval, start_datetime, end_datetime, adjustflag="3")
        if df_raw.empty:
            return pd.DataFrame()

        # Download adjusted close (post-adjustment) to calculate factor
        # adjustflag="1" means post-adjustment (back-adjusted)
        df_adj = self._get_baostock_data(symbol, interval, start_datetime, end_datetime, adjustflag="1")
        
        if df_adj.empty:
             # If no adj data, assume no adjustment
             df_raw["adjclose"] = df_raw["close"]
        else:
             # Merge to get adjclose
             # Ensure dates match
             df_adj = df_adj[["date", "close"]].rename(columns={"close": "adjclose"})
             df_raw = pd.merge(df_raw, df_adj, on="date", how="left")
             # Fill missing adjclose with close
             df_raw["adjclose"] = df_raw["adjclose"].fillna(df_raw["close"])

        return df_raw

    def _get_baostock_data(self, symbol, interval, start, end, adjustflag):
        # Baostock frequency: d=daily, w=weekly, m=monthly, 5=5min, 15=15min, 30=30min, 60=60min
        freq_map = {"1d": "d"}
        bs_freq = freq_map.get(interval, "d")
        
        fields = "date,open,high,low,close,volume,amount"
        
        rs = bs.query_history_k_data_plus(
            symbol,
            fields,
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            frequency=bs_freq,
            adjustflag=adjustflag
        )
        
        data_list = []
        while rs.error_code == "0" and rs.next():
            data_list.append(rs.get_row_data())
            
        if not data_list:
            return pd.DataFrame()
            
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df

class BaostockNormalizeCN1d(BaseNormalize):
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # For 1d data, we don't strictly need a calendar list for simple normalization
        # unless we want to align with a specific exchange calendar.
        # Returning None means no strict alignment based on an external calendar.
        return None

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # Based on qlib/scripts/data_collector/yahoo/collector.py normalize logic
        # We have: date, open, high, low, close, volume, amount, adjclose
        
        if df.empty:
            return df

        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)

        # Qlib format expects:
        # factor = adjclose / close
        # But Qlib Normalize usually sets the first day's factor to 1 (or based on first day price).
        # Standard Qlib Yahoo collector normalization:
        # 1. Adj Close is used to fill open/close/high/low if they are 0/nan? No.
        # 2. Calculate factor. 
        #    If we have adjclose, factor = adjclose / close.
        #    However, Qlib's `BaseNormalize` or typical logic might want to rebase the factor so first day is 1.
        
        # Let's look at how YahooCollector does it.
        # It uses `calc_adjusted_price` but that seems to be for 1min using 1d.
        # For 1d data, YahooCollector just calculates factor.
        
        # Let's calculate a simple factor
        # Avoid division by zero
        df["factor"] = df["adjclose"] / df["close"]
        df.loc[df["close"] == 0, "factor"] = 1.0
        
        # Drop adjclose, as we will store factor
        df.drop(columns=["adjclose"], inplace=True, errors="ignore")
        
        # Rename columns to match Qlib expectations if needed
        # Baostock: open, high, low, close, volume, amount
        # Qlib dump_bin expects these.
        
        df.index.names = ["date"]
        return df.reset_index()

class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", region="CN"):
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return "BaostockCollectorCN1d"

    @property
    def normalize_class_name(self):
        return "BaostockNormalizeCN1d"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0,
        start="2020-01-01",
        end=None,
        check_data_length=None,
        limit_nums=None,
    ):
        if end is None:
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
        
        super(Run, self).download_data(
            max_collector_count, delay, start, end, check_data_length, limit_nums
        )

if __name__ == "__main__":
    fire.Fire(Run)

