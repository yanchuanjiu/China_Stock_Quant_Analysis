#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 Baostock 更新股票数据到 Qlib 格式

功能:
1. 从 Baostock 下载沪深300成分股的最新数据
2. 转换为 Qlib 二进制格式
3. 更新现有的 Qlib 数据目录
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import struct

try:
    import baostock as bs
except ImportError:
    print("请先安装 baostock: pip install baostock")
    sys.exit(1)


def login_baostock():
    """登录 Baostock"""
    lg = bs.login()
    if lg.error_code != '0':
        print(f"❌ 登录失败: {lg.error_msg}")
        return False
    print("✅ Baostock 登录成功")
    return True


def get_stock_list():
    """获取沪深300成分股列表"""
    # 获取最新交易日
    today = datetime.now().strftime('%Y-%m-%d')
    rs = bs.query_trade_dates(start_date='2025-01-01', end_date=today)
    dates = []
    while rs.error_code == '0' and rs.next():
        row = rs.get_row_data()
        if row[1] == '1':  # 交易日
            dates.append(row[0])
    
    if not dates:
        print("❌ 无法获取交易日期")
        return []
    
    latest_date = dates[-1]
    print(f"最新交易日: {latest_date}")
    
    # 获取沪深300成分股
    rs = bs.query_hs300_stocks(date=latest_date)
    stocks = []
    while rs.error_code == '0' and rs.next():
        stocks.append(rs.get_row_data()[1])  # code
    
    print(f"沪深300成分股: {len(stocks)} 只")
    return stocks


def download_stock_data(stock_code, start_date, end_date):
    """下载单只股票数据"""
    rs = bs.query_history_k_data_plus(
        stock_code,
        "date,open,high,low,close,volume,amount,adjustflag,turn,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"  # 前复权
    )
    
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=[
        'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 
        'adjustflag', 'turn', 'pctChg'
    ])
    
    # 转换数据类型
    df['date'] = pd.to_datetime(df['date'])
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def convert_to_qlib_format(df, qlib_symbol):
    """转换为 Qlib 格式"""
    # Qlib 需要的字段
    result = pd.DataFrame()
    result['date'] = df['date']
    result['$open'] = df['open']
    result['$high'] = df['high']
    result['$low'] = df['low']
    result['$close'] = df['close']
    result['$volume'] = df['volume']
    result['$factor'] = 1.0  # 前复权因子设为1
    result['$change'] = df['pctChg'] / 100  # 涨跌幅
    
    result = result.set_index('date')
    result = result.sort_index()
    
    return result


def save_to_qlib_bin(df, output_dir, symbol):
    """保存为 Qlib 二进制格式"""
    output_dir = Path(output_dir)
    
    # 保存日历
    calendar_dir = output_dir / 'calendars'
    calendar_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存特征
    features_dir = output_dir / 'features' / symbol
    features_dir.mkdir(parents=True, exist_ok=True)
    
    for col in df.columns:
        col_name = col.replace('$', '')
        bin_file = features_dir / f'{col_name}.day.bin'
        
        # 写入二进制数据
        values = df[col].values.astype(np.float32)
        with open(bin_file, 'wb') as f:
            f.write(values.tobytes())


def update_qlib_data(start_date='2025-05-15', end_date=None):
    """更新 Qlib 数据"""
    print("=" * 60)
    print("📥 从 Baostock 更新数据")
    print("=" * 60)
    
    if not login_baostock():
        return False
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n更新日期范围: {start_date} ~ {end_date}")
    
    # 获取股票列表
    stocks = get_stock_list()
    if not stocks:
        return False
    
    # 输出目录
    output_dir = Path(__file__).parent / 'data' / 'baostock_update'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载数据
    success_count = 0
    all_dates = set()
    
    print(f"\n开始下载数据...")
    for stock in tqdm(stocks, desc="下载进度"):
        df = download_stock_data(stock, start_date, end_date)
        if df is not None and len(df) > 0:
            # 转换股票代码格式: sh.600000 -> SH600000
            qlib_symbol = stock.replace('.', '').upper()
            
            # 转换格式
            qlib_df = convert_to_qlib_format(df, qlib_symbol)
            
            # 保存 CSV (用于验证)
            csv_file = output_dir / f'{qlib_symbol}.csv'
            qlib_df.to_csv(csv_file)
            
            # 收集日期
            all_dates.update(qlib_df.index.strftime('%Y-%m-%d').tolist())
            
            success_count += 1
    
    # 保存日历
    if all_dates:
        calendar_file = output_dir / 'calendar.txt'
        with open(calendar_file, 'w') as f:
            for date in sorted(all_dates):
                f.write(date + '\n')
    
    bs.logout()
    
    print(f"\n✅ 数据更新完成!")
    print(f"   成功下载: {success_count} 只股票")
    print(f"   交易日数: {len(all_dates)} 天")
    print(f"   保存位置: {output_dir}")
    
    return True


def merge_with_existing_qlib(baostock_dir, qlib_dir):
    """将 Baostock 数据合并到现有 Qlib 数据"""
    baostock_dir = Path(baostock_dir)
    qlib_dir = Path(qlib_dir)
    
    print(f"\n合并数据到: {qlib_dir}")
    
    # 读取现有日历
    existing_calendar_file = qlib_dir / 'calendars' / 'day.txt'
    existing_dates = set()
    if existing_calendar_file.exists():
        with open(existing_calendar_file) as f:
            existing_dates = set(line.strip() for line in f if line.strip())
    
    # 读取新日历
    new_calendar_file = baostock_dir / 'calendar.txt'
    new_dates = set()
    if new_calendar_file.exists():
        with open(new_calendar_file) as f:
            new_dates = set(line.strip() for line in f if line.strip())
    
    # 合并日历
    all_dates = existing_dates | new_dates
    with open(existing_calendar_file, 'w') as f:
        for date in sorted(all_dates):
            f.write(date + '\n')
    
    print(f"   日历更新: {len(existing_dates)} -> {len(all_dates)} 天")
    
    # 这里可以添加更多合并逻辑
    # 由于 Qlib 二进制格式较复杂，建议使用 CSV 数据进行验证
    
    return True


def get_last_update_date():
    """获取上次更新的日期"""
    log_file = Path(__file__).parent / 'data' / 'baostock_update' / 'update_log.txt'
    if log_file.exists():
        with open(log_file) as f:
            lines = f.readlines()
            if lines:
                return lines[-1].strip().split(',')[0]
    return '2025-05-14'


def log_update(date, stock_count):
    """记录更新日志"""
    log_file = Path(__file__).parent / 'data' / 'baostock_update' / 'update_log.txt'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(f"{date},{stock_count},{datetime.now().isoformat()}\n")


def daily_update():
    """每日更新数据"""
    print("=" * 60)
    print("📅 每日数据更新")
    print("=" * 60)
    
    # 获取上次更新日期
    last_date = get_last_update_date()
    print(f"上次更新: {last_date}")
    
    # 计算开始日期 (上次更新日期的下一天)
    from datetime import timedelta
    start_dt = datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)
    start_date = start_dt.strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date > end_date:
        print("✅ 数据已是最新，无需更新")
        return True
    
    print(f"更新范围: {start_date} ~ {end_date}")
    
    # 执行更新
    success = update_qlib_data(start_date, end_date)
    
    if success:
        log_update(end_date, 300)
        print(f"✅ 更新完成，已记录到日志")
    
    return success


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='从 Baostock 更新股票数据')
    parser.add_argument('--start', type=str, default=None, help='开始日期 (默认: 上次更新日期的下一天)')
    parser.add_argument('--end', type=str, default=None, help='结束日期 (默认: 今天)')
    parser.add_argument('--merge', action='store_true', help='合并到现有 Qlib 数据')
    parser.add_argument('--daily', action='store_true', help='每日增量更新模式')
    
    args = parser.parse_args()
    
    if args.daily:
        # 每日增量更新模式
        success = daily_update()
    else:
        # 手动指定日期更新
        start_date = args.start if args.start else '2025-05-15'
        success = update_qlib_data(start_date, args.end)
    
    if success and args.merge:
        qlib_dir = Path.home() / '.qlib' / 'qlib_data' / 'cn_data'
        baostock_dir = Path(__file__).parent / 'data' / 'baostock_update'
        merge_with_existing_qlib(baostock_dir, qlib_dir)
    
    # 输出使用说明
    print("\n" + "=" * 60)
    print("📌 使用说明")
    print("=" * 60)
    print("手动更新: python update_data_baostock.py --start 2025-05-15")
    print("每日更新: python update_data_baostock.py --daily")
    print("定时任务: crontab -e 添加以下行:")
    print("  0 18 * * 1-5 cd /Users/air/QT_China && python update_data_baostock.py --daily")

