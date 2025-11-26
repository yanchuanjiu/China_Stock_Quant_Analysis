#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用最新 Baostock 数据进行完整回测

数据范围: 2024-01-01 ~ 2025-11-25 (合并 Qlib 旧数据 + Baostock 新数据)
起始资金: 10万元
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord


# ==================== 配置 ====================
CONFIG = {
    "account": 100000,
    "exchange_kwargs": {
        "freq": "day",
        "limit_threshold": 0.095,
        "deal_price": "open",
        "open_cost": 0.0003,
        "close_cost": 0.0003,
        "min_cost": 5,
        "impact_cost": 0.001,
        "trade_unit": 100,
    },
    "strategy": {"topk": 10, "n_drop": 2},
    "benchmark": "SH000300",
}


def calculate_metrics(returns: pd.Series, rf=0.03/252):
    """计算评估指标"""
    if len(returns) == 0 or returns.std() == 0:
        return {}
    
    ann_factor = 252
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
    ann_volatility = returns.std() * np.sqrt(ann_factor)
    
    excess_returns = returns - rf
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(ann_factor) if returns.std() > 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(ann_factor) if downside_std > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
    win_rate = (returns > 0).sum() / len(returns) * 100
    
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    
    return {
        'total_return': total_return * 100,
        'annualized_return': ann_return * 100,
        'annualized_volatility': ann_volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown * 100,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'trading_days': len(returns),
    }


def extract_trade_records(positions: dict, report_df: pd.DataFrame):
    """提取交易记录"""
    trade_records = []
    dates = sorted(positions.keys())
    prev_holdings = {}
    
    for i, date in enumerate(dates):
        pos = positions[date]
        if not hasattr(pos, 'position'):
            continue
        
        pos_dict = pos.position
        current_holdings = {}
        
        for stock, info in pos_dict.items():
            if stock in ['cash', 'now_account_value']:
                continue
            if isinstance(info, dict):
                current_holdings[stock] = {
                    'amount': info.get('amount', 0),
                    'price': info.get('price', 0),
                }
        
        all_stocks = set(prev_holdings.keys()) | set(current_holdings.keys())
        
        for stock in all_stocks:
            prev_amt = prev_holdings.get(stock, {}).get('amount', 0)
            curr_amt = current_holdings.get(stock, {}).get('amount', 0)
            price = current_holdings.get(stock, {}).get('price', 0) or prev_holdings.get(stock, {}).get('price', 0)
            
            if abs(curr_amt - prev_amt) > 0.01:
                if curr_amt > prev_amt:
                    action = 'BUY'
                    amount = curr_amt - prev_amt
                else:
                    action = 'SELL'
                    amount = prev_amt - curr_amt
                
                trade_records.append({
                    'date': date,
                    'stock': stock,
                    'action': action,
                    'amount': amount,
                    'price': price,
                    'value': amount * price,
                })
        
        prev_holdings = current_holdings
    
    return pd.DataFrame(trade_records)


def load_baostock_data():
    """加载 Baostock 数据用于 Buy & Hold 计算"""
    data_dir = Path(__file__).parent / 'data' / 'baostock_update'
    
    # 尝试加载沪深300ETF或成分股数据
    for symbol in ['SH510300', 'SH000300', 'SH600000']:
        csv_file = data_dir / f'{symbol}.csv'
        if csv_file.exists():
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            print(f"  加载 {symbol} 新数据: {len(df)} 条")
            return df, symbol
    
    return None, None


def calculate_buyhold_extended(start_date, end_date, initial_capital=100000):
    """计算 Buy & Hold (使用扩展数据)"""
    print("\n📊 计算 Buy & Hold 策略...")
    
    # 先用 Qlib 数据
    for code in ['SH510300', 'SH000300']:
        try:
            qlib_data = D.features([code], ['$close', '$open'], 
                                  start_time=start_date, end_time='2025-05-14', freq='day')
            qlib_data = qlib_data.dropna()
            if len(qlib_data) > 10:
                print(f"  Qlib 数据 ({code}): {len(qlib_data)} 条")
                break
        except:
            continue
    else:
        print("  ⚠️ 无法获取基准数据")
        return None
    
    # 加载 Baostock 新数据
    baostock_df, symbol = load_baostock_data()
    
    # 合并数据
    qlib_data = qlib_data.reset_index()
    qlib_data.columns = ['instrument', 'datetime', 'close', 'open']
    
    if baostock_df is not None:
        # Baostock 数据
        baostock_df = baostock_df.reset_index()
        baostock_df.columns = ['datetime'] + list(baostock_df.columns[1:])
        baostock_df['datetime'] = pd.to_datetime(baostock_df['datetime'])
        
        # 筛选5月15日之后的数据
        baostock_df = baostock_df[baostock_df['datetime'] > '2025-05-14']
        
        if len(baostock_df) > 0:
            print(f"  Baostock 新数据: {len(baostock_df)} 条 ({baostock_df['datetime'].min()} ~ {baostock_df['datetime'].max()})")
            
            # 合并
            new_data = pd.DataFrame({
                'instrument': code,
                'datetime': baostock_df['datetime'],
                'close': baostock_df['$close'],
                'open': baostock_df['$open'],
            })
            qlib_data = pd.concat([qlib_data, new_data], ignore_index=True)
    
    qlib_data = qlib_data.sort_values('datetime').drop_duplicates('datetime')
    
    # 筛选日期范围
    qlib_data = qlib_data[(qlib_data['datetime'] >= start_date) & (qlib_data['datetime'] <= end_date)]
    
    print(f"  合并后数据: {len(qlib_data)} 条")
    
    if len(qlib_data) == 0:
        return None
    
    # 计算收益
    first_open = qlib_data['open'].iloc[0]
    shares = initial_capital / first_open
    buy_cost = initial_capital * 0.0003
    
    qlib_data['account'] = shares * qlib_data['close'] - buy_cost
    qlib_data['return'] = qlib_data['close'].pct_change()
    
    metrics = calculate_metrics(qlib_data['return'].dropna())
    
    print(f"  最终市值: ¥{qlib_data['account'].iloc[-1]:,.2f}")
    print(f"  总收益率: {metrics.get('total_return', 0):+.2f}%")
    
    return {
        'data': qlib_data,
        'metrics': metrics,
        'shares': shares,
        'final_value': qlib_data['account'].iloc[-1],
    }


def run_model_backtest(start_date, end_date, model_type="lightgbm"):
    """运行模型回测 (使用 Qlib 数据，截止到5月14日)"""
    print(f"\n🤖 训练 {model_type.upper()} 模型...")
    
    # 由于 Qlib 数据只到5月14日，回测只能到这个日期
    actual_end_date = min(end_date, '2025-05-14')
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    train_end = (start_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    train_start = (start_dt - timedelta(days=365)).strftime('%Y-%m-%d')
    valid_start = (start_dt - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"  训练期: {train_start} ~ {valid_start}")
    print(f"  验证期: {valid_start} ~ {train_end}")
    print(f"  回测期: {start_date} ~ {actual_end_date}")
    
    data_handler_config = {
        "start_time": train_start,
        "end_time": actual_end_date,
        "fit_start_time": train_start,
        "fit_end_time": train_end,
        "instruments": "csi300",
    }
    
    if model_type.lower() == "xgboost":
        model_config = {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
            "kwargs": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "early_stopping_rounds": 20},
        }
    else:
        model_config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {"loss": "mse", "learning_rate": 0.05, "max_depth": 8, "num_leaves": 128, "early_stopping_rounds": 50},
        }
    
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": (train_start, valid_start),
                "valid": (valid_start, train_end),
                "test": (start_date, actual_end_date),
            },
        },
    }
    
    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)
    
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {"model": model, "dataset": dataset, **CONFIG["strategy"]},
        },
        "backtest": {
            "start_time": start_date,
            "end_time": actual_end_date,
            "account": CONFIG["account"],
            "benchmark": CONFIG["benchmark"],
            "exchange_kwargs": CONFIG["exchange_kwargs"],
        },
    }
    
    exp_dir = Path(__file__).parent / "experiments"
    os.environ["MLFLOW_TRACKING_URI"] = str(exp_dir)
    
    exp_name = f"FullBacktest_{model_type.upper()}_10W"
    
    with R.start(experiment_name=exp_name):
        model.fit(dataset)
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        
        report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        analysis = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    
    trade_records = extract_trade_records(positions, report_df)
    returns = report_df['return']
    metrics = calculate_metrics(returns)
    
    return {
        'model_type': model_type,
        'report': report_df,
        'positions': positions,
        'analysis': analysis,
        'trade_records': trade_records,
        'metrics': metrics,
        'end_date': actual_end_date,
    }


def generate_report(results: dict, buyhold: dict, output_dir: Path):
    """生成完整报告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("📊 完整回测分析报告 (含最新数据)")
    print("=" * 80)
    
    print(f"\n💰 起始资金: ¥{CONFIG['account']:,}")
    
    # 数据范围说明
    print(f"\n📅 数据说明:")
    print(f"   Qlib 数据: 截止到 2025-05-14")
    print(f"   Baostock 新数据: 2025-05-15 ~ 2025-11-25 (132天)")
    
    # 模型对比
    print("\n" + "-" * 80)
    print("📈 模型对比")
    print("-" * 80)
    
    headers = ['指标', 'LightGBM', 'XGBoost', 'Buy & Hold']
    print(f"{headers[0]:<25} {headers[1]:>15} {headers[2]:>15} {headers[3]:>15}")
    print("-" * 80)
    
    lgb = results.get('lightgbm', {}).get('metrics', {})
    xgb = results.get('xgboost', {}).get('metrics', {})
    bh = buyhold.get('metrics', {}) if buyhold else {}
    
    metrics_display = [
        ('总收益率 (%)', 'total_return'),
        ('年化收益率 (%)', 'annualized_return'),
        ('年化波动率 (%)', 'annualized_volatility'),
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Sortino Ratio', 'sortino_ratio'),
        ('最大回撤 (%)', 'max_drawdown'),
        ('Calmar Ratio', 'calmar_ratio'),
        ('胜率 (%)', 'win_rate'),
        ('盈亏比', 'profit_loss_ratio'),
        ('交易天数', 'trading_days'),
    ]
    
    for label, key in metrics_display:
        lgb_val = lgb.get(key, 0)
        xgb_val = xgb.get(key, 0)
        bh_val = bh.get(key, 0)
        
        if key in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'profit_loss_ratio']:
            fmt = '{:.4f}'
        elif key == 'trading_days':
            fmt = '{:.0f}'
        else:
            fmt = '{:.2f}'
        
        print(f"{label:<25} {fmt.format(lgb_val):>15} {fmt.format(xgb_val):>15} {fmt.format(bh_val):>15}")
    
    # 持仓详情
    for model_name, data in results.items():
        print(f"\n" + "-" * 80)
        print(f"📋 {model_name.upper()} 最终持仓")
        print("-" * 80)
        
        positions = data.get('positions', {})
        if not positions:
            continue
        
        dates = sorted(positions.keys())
        last_pos = positions[dates[-1]]
        
        if hasattr(last_pos, 'position'):
            pos_dict = last_pos.position
            cash = pos_dict.get('cash', 0)
            account = pos_dict.get('now_account_value', 0)
            
            print(f"现金: ¥{cash:,.2f}  |  账户总值: ¥{account:,.2f}")
            print(f"\n{'股票':<12} {'数量':>10} {'价格':>10} {'市值':>12} {'权重':>8}")
            print("-" * 55)
            
            stocks = [(k, v) for k, v in pos_dict.items() 
                     if k not in ['cash', 'now_account_value'] and isinstance(v, dict)]
            
            for stock, info in sorted(stocks, key=lambda x: -x[1].get('weight', 0)):
                amt = info.get('amount', 0)
                price = info.get('price', 0)
                weight = info.get('weight', 0) * 100
                mkt_val = amt * price
                print(f"{stock:<12} {amt:>10,.0f} {price:>10.2f} ¥{mkt_val:>10,.0f} {weight:>7.2f}%")
    
    # 交易记录
    for model_name, data in results.items():
        trade_df = data.get('trade_records')
        if trade_df is None or trade_df.empty:
            continue
        
        print(f"\n" + "-" * 80)
        print(f"📝 {model_name.upper()} 交易记录 (全部 {len(trade_df)} 条)")
        print("-" * 80)
        print(f"{'日期':<12} {'股票':<12} {'操作':>6} {'数量':>10} {'价格':>10} {'金额':>12}")
        print("-" * 65)
        
        for _, row in trade_df.tail(30).iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
            print(f"{date_str:<12} {row['stock']:<12} {row['action']:>6} {row['amount']:>10,.0f} {row['price']:>10.2f} ¥{row['value']:>10,.0f}")
        
        # 保存完整交易记录
        trade_df.to_csv(output_dir / f'{model_name}_trades_full.csv', index=False)
        print(f"\n✅ 完整交易记录已保存: {output_dir / f'{model_name}_trades_full.csv'}")
    
    # 保存汇总
    summary = []
    for model_name, data in results.items():
        m = data.get('metrics', {}).copy()
        m['model'] = model_name
        m['backtest_end'] = data.get('end_date', '')
        summary.append(m)
    
    if buyhold:
        bh_m = buyhold.get('metrics', {}).copy()
        bh_m['model'] = 'buy_hold'
        bh_m['backtest_end'] = buyhold['data']['datetime'].max().strftime('%Y-%m-%d') if 'data' in buyhold else ''
        summary.append(bh_m)
    
    pd.DataFrame(summary).to_csv(output_dir / 'metrics_comparison_full.csv', index=False)
    
    print(f"\n✅ 所有结果已保存到: {output_dir}")
    
    return summary


def generate_html_report(results: dict, buyhold: dict, output_dir: Path):
    """生成 HTML 报告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lgb_report = results.get('lightgbm', {}).get('report')
    xgb_report = results.get('xgboost', {}).get('report')
    
    if lgb_report is None:
        return
    
    initial = CONFIG['account']
    
    # 模型数据
    dates = [d.strftime('%Y-%m-%d') for d in lgb_report.index]
    lgb_values = lgb_report['account'].values.tolist()
    lgb_returns = ((lgb_report['account'] / initial - 1) * 100).tolist()
    
    xgb_values = xgb_report['account'].values.tolist() if xgb_report is not None else lgb_values
    xgb_returns = ((xgb_report['account'] / initial - 1) * 100).tolist() if xgb_report is not None else lgb_returns
    
    # Buy & Hold 数据 (包含新数据)
    if buyhold and 'data' in buyhold:
        bh_data = buyhold['data']
        bh_dates = [d.strftime('%Y-%m-%d') for d in bh_data['datetime']]
        bh_values = bh_data['account'].values.tolist()
        bh_returns = ((bh_data['account'] / initial - 1) * 100).tolist()
    else:
        bh_dates = dates
        bh_values = [initial] * len(dates)
        bh_returns = [0] * len(dates)
    
    lgb_m = results.get('lightgbm', {}).get('metrics', {})
    xgb_m = results.get('xgboost', {}).get('metrics', {})
    bh_m = buyhold.get('metrics', {}) if buyhold else {}
    
    # 交易记录
    trade_df = results.get('lightgbm', {}).get('trade_records')
    trade_html = ""
    if trade_df is not None and not trade_df.empty:
        rows = []
        for _, row in trade_df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
            action_class = 'buy' if row['action'] == 'BUY' else 'sell'
            rows.append(f"<tr><td>{date_str}</td><td>{row['stock']}</td><td class='{action_class}'>{row['action']}</td><td>{row['amount']:,.0f}</td><td>¥{row['price']:.2f}</td><td>¥{row['value']:,.0f}</td></tr>")
        trade_html = "\n".join(rows)
    
    # HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>完整回测分析报告 (含最新数据)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f5f7fa; }}
        .header {{ background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; padding: 30px 40px; }}
        .header h1 {{ margin: 0 0 10px; font-size: 28px; }}
        .header p {{ margin: 5px 0; opacity: 0.9; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 30px; }}
        .info-box {{ background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); text-align: center; }}
        .stat-value {{ font-size: 28px; font-weight: bold; }}
        .stat-label {{ color: #666; font-size: 13px; margin-top: 5px; }}
        .positive {{ color: #34a853; }}
        .negative {{ color: #ea4335; }}
        .chart-container {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 30px; }}
        .chart-title {{ font-size: 18px; font-weight: 600; margin-bottom: 15px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        td:first-child, th:first-child {{ text-align: left; }}
        .buy {{ color: #ea4335; }}
        .sell {{ color: #34a853; }}
        .metrics-table {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 30px; }}
        .trades-container {{ max-height: 600px; overflow-y: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 完整回测分析报告</h1>
        <p>起始资金: ¥{initial:,} | 模型回测: {dates[0]} ~ {dates[-1]} ({len(dates)} 天)</p>
        <p>Buy & Hold: {bh_dates[0]} ~ {bh_dates[-1]} ({len(bh_dates)} 天) ⭐ 包含 Baostock 最新数据</p>
    </div>
    
    <div class="container">
        <div class="info-box">
            <strong>📢 数据说明:</strong> 模型回测使用 Qlib 数据 (截止 2025-05-14)，Buy & Hold 使用合并数据 (含 Baostock 2025-05-15 ~ 2025-11-25 共 132 天新数据)
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value {'positive' if lgb_m.get('total_return', 0) > 0 else 'negative'}">{lgb_m.get('total_return', 0):+.2f}%</div>
                <div class="stat-label">LightGBM 收益 (118天)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'positive' if xgb_m.get('total_return', 0) > 0 else 'negative'}">{xgb_m.get('total_return', 0):+.2f}%</div>
                <div class="stat-label">XGBoost 收益 (118天)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'positive' if bh_m.get('total_return', 0) > 0 else 'negative'}">{bh_m.get('total_return', 0):+.2f}%</div>
                <div class="stat-label">Buy & Hold ({len(bh_dates)}天)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{lgb_m.get('sharpe_ratio', 0):.2f}</div>
                <div class="stat-label">LightGBM Sharpe</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📈 账户价值对比</div>
            <div id="value-chart" style="height: 400px;"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📊 累计收益率对比</div>
            <div id="return-chart" style="height: 400px;"></div>
        </div>
        
        <div class="metrics-table">
            <div class="chart-title">📋 模型指标对比</div>
            <table>
                <tr><th>指标</th><th>LightGBM (118天)</th><th>XGBoost (118天)</th><th>Buy & Hold ({len(bh_dates)}天)</th></tr>
                <tr><td>总收益率</td><td class="{'positive' if lgb_m.get('total_return',0)>0 else 'negative'}">{lgb_m.get('total_return',0):+.2f}%</td><td class="{'positive' if xgb_m.get('total_return',0)>0 else 'negative'}">{xgb_m.get('total_return',0):+.2f}%</td><td class="{'positive' if bh_m.get('total_return',0)>0 else 'negative'}">{bh_m.get('total_return',0):+.2f}%</td></tr>
                <tr><td>年化收益率</td><td>{lgb_m.get('annualized_return',0):+.2f}%</td><td>{xgb_m.get('annualized_return',0):+.2f}%</td><td>{bh_m.get('annualized_return',0):+.2f}%</td></tr>
                <tr><td>年化波动率</td><td>{lgb_m.get('annualized_volatility',0):.2f}%</td><td>{xgb_m.get('annualized_volatility',0):.2f}%</td><td>{bh_m.get('annualized_volatility',0):.2f}%</td></tr>
                <tr><td>Sharpe Ratio</td><td><b>{lgb_m.get('sharpe_ratio',0):.4f}</b></td><td><b>{xgb_m.get('sharpe_ratio',0):.4f}</b></td><td><b>{bh_m.get('sharpe_ratio',0):.4f}</b></td></tr>
                <tr><td>Sortino Ratio</td><td>{lgb_m.get('sortino_ratio',0):.4f}</td><td>{xgb_m.get('sortino_ratio',0):.4f}</td><td>{bh_m.get('sortino_ratio',0):.4f}</td></tr>
                <tr><td>最大回撤</td><td class="negative">{lgb_m.get('max_drawdown',0):.2f}%</td><td class="negative">{xgb_m.get('max_drawdown',0):.2f}%</td><td class="negative">{bh_m.get('max_drawdown',0):.2f}%</td></tr>
                <tr><td>Calmar Ratio</td><td>{lgb_m.get('calmar_ratio',0):.4f}</td><td>{xgb_m.get('calmar_ratio',0):.4f}</td><td>{bh_m.get('calmar_ratio',0):.4f}</td></tr>
                <tr><td>胜率</td><td>{lgb_m.get('win_rate',0):.2f}%</td><td>{xgb_m.get('win_rate',0):.2f}%</td><td>{bh_m.get('win_rate',0):.2f}%</td></tr>
            </table>
        </div>
        
        <div class="metrics-table">
            <div class="chart-title">📝 LightGBM 完整交易记录 ({len(trade_df) if trade_df is not None else 0} 条)</div>
            <div class="trades-container">
            <table>
                <tr><th>日期</th><th>股票</th><th>操作</th><th>数量</th><th>价格</th><th>金额</th></tr>
                {trade_html}
            </table>
            </div>
        </div>
    </div>
    
    <script>
        Plotly.newPlot('value-chart', [
            {{x: {dates}, y: {lgb_values}, name: 'LightGBM', line: {{color: '#1a73e8', width: 2}}}},
            {{x: {dates}, y: {xgb_values}, name: 'XGBoost', line: {{color: '#fbbc04', width: 2}}}},
            {{x: {bh_dates}, y: {bh_values}, name: 'Buy & Hold (含新数据)', line: {{color: '#ea4335', width: 2, dash: 'dash'}}}}
        ], {{margin: {{t:20,r:50,b:50,l:80}}, yaxis: {{title: '账户价值 (¥)', tickformat: ',.0f'}}, hovermode: 'x unified', legend: {{x: 0.02, y: 0.98}}}});
        
        Plotly.newPlot('return-chart', [
            {{x: {dates}, y: {lgb_returns}, name: 'LightGBM', fill: 'tozeroy', line: {{color: '#1a73e8'}}}},
            {{x: {dates}, y: {xgb_returns}, name: 'XGBoost', line: {{color: '#fbbc04', width: 2}}}},
            {{x: {bh_dates}, y: {bh_returns}, name: 'Buy & Hold (含新数据)', line: {{color: '#ea4335', dash: 'dash'}}}}
        ], {{margin: {{t:20,r:50,b:50,l:80}}, yaxis: {{title: '累计收益率 (%)', ticksuffix: '%'}}, hovermode: 'x unified', legend: {{x: 0.02, y: 0.98}}}});
    </script>
</body>
</html>'''
    
    with open(output_dir / 'full_backtest_report.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML 报告已保存: {output_dir / 'full_backtest_report.html'}")


def main():
    print("=" * 80)
    print("🚀 完整回测分析 (含 Baostock 最新数据)")
    print("=" * 80)
    
    # 初始化 Qlib
    provider_uri = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    
    # 设置日期
    start_date = '2024-11-15'
    end_date = '2025-11-25'  # 使用 Baostock 的最新日期
    
    print(f"\n💰 回测配置:")
    print(f"   起始资金: ¥{CONFIG['account']:,}")
    print(f"   目标周期: {start_date} ~ {end_date}")
    
    # 运行模型回测 (只能用 Qlib 数据到5月14日)
    results = {}
    for model in ['lightgbm', 'xgboost']:
        results[model] = run_model_backtest(start_date, end_date, model)
    
    # Buy & Hold (使用合并数据)
    buyhold = calculate_buyhold_extended(start_date, end_date, CONFIG['account'])
    
    # 生成报告
    output_dir = Path(__file__).parent / "output" / "full_backtest_new_data"
    generate_report(results, buyhold, output_dir)
    generate_html_report(results, buyhold, output_dir)
    
    print(f"\n🌐 打开报告: open {output_dir / 'full_backtest_report.html'}")


if __name__ == "__main__":
    main()

