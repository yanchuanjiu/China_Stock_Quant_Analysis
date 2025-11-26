#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整回测分析脚本

功能:
1. 支持从 Baostock 更新最新数据
2. 多模型对比 (XGBoost, LightGBM)
3. Buy & Hold 策略对比
4. 完整的评估指标 (Sharpe Ratio, Sortino, Calmar 等)
5. 详细交易记录 (每日买卖明细)

配置:
- 起始资金: 10万元
- 交易单位: 100股/手
- 手续费: 万分之三
- 滑点: 0.1%
- 成交价: 开盘价
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
    "account": 100000,  # 起始资金 10万
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


def update_data_from_baostock(start_date='2025-05-15', end_date=None):
    """从 Baostock 更新最新数据"""
    try:
        import baostock as bs
    except ImportError:
        print("❌ Baostock 未安装，请运行: pip install baostock")
        return False
    
    print("=" * 60)
    print("📥 从 Baostock 更新数据")
    print("=" * 60)
    
    lg = bs.login()
    if lg.error_code != '0':
        print(f"❌ 登录失败: {lg.error_msg}")
        return False
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"更新日期范围: {start_date} ~ {end_date}")
    
    # 获取沪深300成分股
    rs = bs.query_hs300_stocks()
    stocks = []
    while rs.error_code == '0' and rs.next():
        stocks.append(rs.get_row_data()[1])
    
    print(f"获取到 {len(stocks)} 只股票")
    
    # 创建输出目录
    output_dir = Path(__file__).parent / "data" / "baostock_update"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for i, stock in enumerate(stocks):
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(stocks)}")
        
        rs = bs.query_history_k_data_plus(
            stock,
            "date,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"
        )
        
        data = []
        while rs.error_code == '0' and rs.next():
            data.append(rs.get_row_data())
        
        if data:
            df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
            # 转换格式
            symbol = stock.replace(".", "").upper()
            df.to_csv(output_dir / f"{symbol}.csv", index=False)
            success_count += 1
    
    bs.logout()
    print(f"✅ 成功更新 {success_count} 只股票的数据")
    print(f"   保存位置: {output_dir}")
    
    return True


def calculate_metrics(returns: pd.Series, rf=0.03/252):
    """计算完整的评估指标"""
    if len(returns) == 0 or returns.std() == 0:
        return {}
    
    # 年化参数
    ann_factor = 252
    
    # 基本统计
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
    ann_volatility = returns.std() * np.sqrt(ann_factor)
    
    # Sharpe Ratio
    excess_returns = returns - rf
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(ann_factor) if returns.std() > 0 else 0
    
    # Sortino Ratio (只考虑下行波动)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(ann_factor) if downside_std > 0 else 0
    
    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win Rate
    win_rate = (returns > 0).sum() / len(returns) * 100
    
    # Profit/Loss Ratio
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
    """提取详细的交易记录"""
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
        
        # 比较持仓变化
        all_stocks = set(prev_holdings.keys()) | set(current_holdings.keys())
        
        for stock in all_stocks:
            prev_amt = prev_holdings.get(stock, {}).get('amount', 0)
            curr_amt = current_holdings.get(stock, {}).get('amount', 0)
            price = current_holdings.get(stock, {}).get('price', 0) or prev_holdings.get(stock, {}).get('price', 0)
            
            if abs(curr_amt - prev_amt) > 0.01:  # 有变化
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


def run_model_backtest(start_date, end_date, model_type="lightgbm"):
    """运行单个模型的回测"""
    print(f"\n🤖 训练 {model_type.upper()} 模型...")
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    train_end = (start_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    train_start = (start_dt - timedelta(days=365)).strftime('%Y-%m-%d')
    valid_start = (start_dt - timedelta(days=90)).strftime('%Y-%m-%d')
    
    data_handler_config = {
        "start_time": train_start,
        "end_time": end_date,
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
        exp_name = f"Analysis_{model_type.upper()}_10W"
    else:
        model_config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {"loss": "mse", "learning_rate": 0.05, "max_depth": 8, "num_leaves": 128, "early_stopping_rounds": 50},
        }
        exp_name = f"Analysis_{model_type.upper()}_10W"
    
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
                "test": (start_date, end_date),
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
            "end_time": end_date,
            "account": CONFIG["account"],
            "benchmark": CONFIG["benchmark"],
            "exchange_kwargs": CONFIG["exchange_kwargs"],
        },
    }
    
    exp_dir = Path(__file__).parent / "experiments"
    os.environ["MLFLOW_TRACKING_URI"] = str(exp_dir)
    
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
    
    # 提取交易记录
    trade_records = extract_trade_records(positions, report_df)
    
    # 计算指标
    returns = report_df['return']
    metrics = calculate_metrics(returns)
    
    return {
        'model_type': model_type,
        'report': report_df,
        'positions': positions,
        'analysis': analysis,
        'trade_records': trade_records,
        'metrics': metrics,
    }


def calculate_buyhold(start_date, end_date, initial_capital=100000):
    """计算 Buy & Hold 策略"""
    print("\n📊 计算 Buy & Hold 策略...")
    
    for code in ['SH510300', 'SH000300']:
        try:
            data = D.features([code], ['$close', '$open'], start_time=start_date, end_time=end_date, freq='day')
            data = data.dropna()
            if len(data) > 10:
                break
        except:
            continue
    else:
        return None
    
    data = data.reset_index()
    data.columns = ['instrument', 'datetime', 'close', 'open']
    data = data.sort_values('datetime')
    
    first_open = data['open'].iloc[0]
    shares = initial_capital / first_open
    buy_cost = initial_capital * 0.0003
    
    data['account'] = shares * data['close'] - buy_cost
    data['return'] = data['close'].pct_change()
    
    metrics = calculate_metrics(data['return'].dropna())
    
    return {
        'data': data,
        'metrics': metrics,
        'shares': shares,
        'final_value': data['account'].iloc[-1],
    }


def generate_detailed_report(results: dict, buyhold: dict, output_dir: Path):
    """生成详细报告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("📊 完整回测分析报告")
    print("=" * 80)
    
    print(f"\n💰 起始资金: ¥{CONFIG['account']:,}")
    
    # 模型对比表
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
    ]
    
    for label, key in metrics_display:
        lgb_val = lgb.get(key, 0)
        xgb_val = xgb.get(key, 0)
        bh_val = bh.get(key, 0)
        
        if key in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'profit_loss_ratio']:
            fmt = '{:.4f}'
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
        print(f"📝 {model_name.upper()} 交易记录 (最近30条)")
        print("-" * 80)
        print(f"{'日期':<12} {'股票':<12} {'操作':>6} {'数量':>10} {'价格':>10} {'金额':>12}")
        print("-" * 65)
        
        for _, row in trade_df.tail(30).iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
            print(f"{date_str:<12} {row['stock']:<12} {row['action']:>6} {row['amount']:>10,.0f} {row['price']:>10.2f} ¥{row['value']:>10,.0f}")
        
        # 保存完整交易记录
        trade_df.to_csv(output_dir / f'{model_name}_trades.csv', index=False)
        print(f"\n✅ 完整交易记录已保存: {output_dir / f'{model_name}_trades.csv'}")
    
    # 保存汇总
    summary = []
    for model_name, data in results.items():
        m = data.get('metrics', {})
        m['model'] = model_name
        summary.append(m)
    
    if buyhold:
        bh_m = buyhold.get('metrics', {})
        bh_m['model'] = 'buy_hold'
        summary.append(bh_m)
    
    pd.DataFrame(summary).to_csv(output_dir / 'metrics_comparison.csv', index=False)
    
    print(f"\n✅ 所有结果已保存到: {output_dir}")
    
    return summary


def generate_html_report(results: dict, buyhold: dict, output_dir: Path):
    """生成 HTML 可视化报告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    lgb_report = results.get('lightgbm', {}).get('report')
    xgb_report = results.get('xgboost', {}).get('report')
    
    if lgb_report is None:
        return
    
    initial = CONFIG['account']
    dates = [d.strftime('%Y-%m-%d') for d in lgb_report.index]
    
    lgb_values = lgb_report['account'].values.tolist()
    lgb_returns = ((lgb_report['account'] / initial - 1) * 100).tolist()
    
    xgb_values = xgb_report['account'].values.tolist() if xgb_report is not None else lgb_values
    xgb_returns = ((xgb_report['account'] / initial - 1) * 100).tolist() if xgb_report is not None else lgb_returns
    
    bh_values = buyhold['data']['account'].values.tolist() if buyhold else [initial] * len(dates)
    bh_returns = ((buyhold['data']['account'] / initial - 1) * 100).tolist() if buyhold else [0] * len(dates)
    
    lgb_m = results.get('lightgbm', {}).get('metrics', {})
    xgb_m = results.get('xgboost', {}).get('metrics', {})
    bh_m = buyhold.get('metrics', {}) if buyhold else {}
    
    # 交易记录表格
    trade_df = results.get('lightgbm', {}).get('trade_records')
    trade_html = ""
    if trade_df is not None and not trade_df.empty:
        rows = []
        for _, row in trade_df.tail(50).iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
            action_class = 'buy' if row['action'] == 'BUY' else 'sell'
            rows.append(f"<tr><td>{date_str}</td><td>{row['stock']}</td><td class='{action_class}'>{row['action']}</td><td>{row['amount']:,.0f}</td><td>¥{row['price']:.2f}</td><td>¥{row['value']:,.0f}</td></tr>")
        trade_html = "\n".join(rows)
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>完整回测分析报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f5f7fa; }}
        .header {{ background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; padding: 30px 40px; }}
        .header h1 {{ margin: 0 0 10px; font-size: 28px; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 30px; }}
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
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 完整回测分析报告</h1>
        <p>起始资金: ¥{initial:,} | 回测周期: {dates[0]} ~ {dates[-1]} | 交易天数: {len(dates)}</p>
    </div>
    
    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value {'positive' if lgb_m.get('total_return', 0) > 0 else 'negative'}">{lgb_m.get('total_return', 0):+.2f}%</div>
                <div class="stat-label">LightGBM 总收益</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'positive' if xgb_m.get('total_return', 0) > 0 else 'negative'}">{xgb_m.get('total_return', 0):+.2f}%</div>
                <div class="stat-label">XGBoost 总收益</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'positive' if bh_m.get('total_return', 0) > 0 else 'negative'}">{bh_m.get('total_return', 0):+.2f}%</div>
                <div class="stat-label">Buy & Hold 收益</div>
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
                <tr><th>指标</th><th>LightGBM</th><th>XGBoost</th><th>Buy & Hold</th></tr>
                <tr><td>总收益率</td><td class="{'positive' if lgb_m.get('total_return',0)>0 else 'negative'}">{lgb_m.get('total_return',0):+.2f}%</td><td class="{'positive' if xgb_m.get('total_return',0)>0 else 'negative'}">{xgb_m.get('total_return',0):+.2f}%</td><td class="{'positive' if bh_m.get('total_return',0)>0 else 'negative'}">{bh_m.get('total_return',0):+.2f}%</td></tr>
                <tr><td>年化收益率</td><td>{lgb_m.get('annualized_return',0):+.2f}%</td><td>{xgb_m.get('annualized_return',0):+.2f}%</td><td>{bh_m.get('annualized_return',0):+.2f}%</td></tr>
                <tr><td>年化波动率</td><td>{lgb_m.get('annualized_volatility',0):.2f}%</td><td>{xgb_m.get('annualized_volatility',0):.2f}%</td><td>{bh_m.get('annualized_volatility',0):.2f}%</td></tr>
                <tr><td>Sharpe Ratio</td><td><b>{lgb_m.get('sharpe_ratio',0):.4f}</b></td><td><b>{xgb_m.get('sharpe_ratio',0):.4f}</b></td><td><b>{bh_m.get('sharpe_ratio',0):.4f}</b></td></tr>
                <tr><td>Sortino Ratio</td><td>{lgb_m.get('sortino_ratio',0):.4f}</td><td>{xgb_m.get('sortino_ratio',0):.4f}</td><td>{bh_m.get('sortino_ratio',0):.4f}</td></tr>
                <tr><td>最大回撤</td><td class="negative">{lgb_m.get('max_drawdown',0):.2f}%</td><td class="negative">{xgb_m.get('max_drawdown',0):.2f}%</td><td class="negative">{bh_m.get('max_drawdown',0):.2f}%</td></tr>
                <tr><td>Calmar Ratio</td><td>{lgb_m.get('calmar_ratio',0):.4f}</td><td>{xgb_m.get('calmar_ratio',0):.4f}</td><td>{bh_m.get('calmar_ratio',0):.4f}</td></tr>
                <tr><td>胜率</td><td>{lgb_m.get('win_rate',0):.2f}%</td><td>{xgb_m.get('win_rate',0):.2f}%</td><td>{bh_m.get('win_rate',0):.2f}%</td></tr>
                <tr><td>盈亏比</td><td>{lgb_m.get('profit_loss_ratio',0):.4f}</td><td>{xgb_m.get('profit_loss_ratio',0):.4f}</td><td>{bh_m.get('profit_loss_ratio',0):.4f}</td></tr>
            </table>
        </div>
        
        <div class="metrics-table">
            <div class="chart-title">📝 交易记录 (最近50条)</div>
            <table>
                <tr><th>日期</th><th>股票</th><th>操作</th><th>数量</th><th>价格</th><th>金额</th></tr>
                {trade_html}
            </table>
        </div>
    </div>
    
    <script>
        Plotly.newPlot('value-chart', [
            {{x: {dates}, y: {lgb_values}, name: 'LightGBM', line: {{color: '#1a73e8', width: 2}}}},
            {{x: {dates}, y: {xgb_values}, name: 'XGBoost', line: {{color: '#fbbc04', width: 2}}}},
            {{x: {dates}, y: {bh_values}, name: 'Buy & Hold', line: {{color: '#ea4335', width: 2, dash: 'dash'}}}}
        ], {{margin: {{t:20,r:50,b:50,l:80}}, yaxis: {{title: '账户价值 (¥)', tickformat: ',.0f'}}, hovermode: 'x unified'}});
        
        Plotly.newPlot('return-chart', [
            {{x: {dates}, y: {lgb_returns}, name: 'LightGBM', fill: 'tozeroy', line: {{color: '#1a73e8'}}}},
            {{x: {dates}, y: {xgb_returns}, name: 'XGBoost', line: {{color: '#fbbc04', width: 2}}}},
            {{x: {dates}, y: {bh_returns}, name: 'Buy & Hold', line: {{color: '#ea4335', dash: 'dash'}}}}
        ], {{margin: {{t:20,r:50,b:50,l:80}}, yaxis: {{title: '累计收益率 (%)', ticksuffix: '%'}}, hovermode: 'x unified'}});
    </script>
</body>
</html>'''
    
    with open(output_dir / 'full_analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML 报告已保存: {output_dir / 'full_analysis_report.html'}")


def main():
    print("=" * 70)
    print("🚀 完整回测分析")
    print("=" * 70)
    
    # 初始化 Qlib
    provider_uri = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    
    # 获取数据范围
    data = D.features(['SH600000'], ['$close'], start_time='2024-01-01', end_time='2025-12-31', freq='day')
    data = data.dropna()
    last_date = data.index[-1][1]
    
    print(f"📊 数据截止日期: {last_date.strftime('%Y-%m-%d')}")
    
    # 设置回测日期
    end_date = last_date.strftime('%Y-%m-%d')
    start_date = '2024-11-15'
    
    print(f"\n💰 回测配置:")
    print(f"   起始资金: ¥{CONFIG['account']:,}")
    print(f"   回测周期: {start_date} ~ {end_date}")
    
    # 运行模型回测
    results = {}
    for model in ['lightgbm', 'xgboost']:
        results[model] = run_model_backtest(start_date, end_date, model)
    
    # Buy & Hold
    buyhold = calculate_buyhold(start_date, end_date, CONFIG['account'])
    
    # 生成报告
    output_dir = Path(__file__).parent / "output" / "full_analysis"
    generate_detailed_report(results, buyhold, output_dir)
    generate_html_report(results, buyhold, output_dir)
    
    print(f"\n🌐 打开报告: open {output_dir / 'full_analysis_report.html'}")


if __name__ == "__main__":
    main()

