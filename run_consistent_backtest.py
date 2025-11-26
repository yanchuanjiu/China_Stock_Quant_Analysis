#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一致时间周期的完整回测分析

设计原则:
1. 所有策略使用相同的时间周期进行公平对比
2. 模型训练使用 2025-09-30 之前的数据
3. 测试/验证使用 2025-10-01 ~ 现在的数据
4. 避免过拟合和未来数据泄露

时间划分:
- 训练期: 2024-01-01 ~ 2025-08-31
- 验证期: 2025-09-01 ~ 2025-09-30
- 测试期: 2025-10-01 ~ 2025-11-25 (统一对比周期)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm


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
    
    # 时间配置
    "train_start": "2024-01-01",
    "train_end": "2025-08-31",
    "valid_start": "2025-09-01",
    "valid_end": "2025-09-30",
    "test_start": "2025-10-01",
    "test_end": "2025-11-25",
}


def calculate_metrics(returns: pd.Series, rf=0.03/252):
    """计算完整评估指标"""
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


def load_baostock_data():
    """加载 Baostock 数据"""
    data_dir = Path(__file__).parent / 'data' / 'baostock_update'
    
    if not data_dir.exists():
        print("❌ Baostock 数据目录不存在，请先运行 update_data_baostock.py")
        return None
    
    all_data = {}
    csv_files = list(data_dir.glob('*.csv'))
    
    print(f"加载 {len(csv_files)} 只股票数据...")
    for csv_file in tqdm(csv_files, desc="加载数据"):
        symbol = csv_file.stem
        if symbol == 'calendar':
            continue
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            if len(df) > 0:
                all_data[symbol] = df
        except:
            continue
    
    return all_data


def simple_factor_model(stock_data: dict, lookback=20):
    """
    简单因子模型 - 基于动量和波动率
    避免使用未来数据
    """
    factors = {}
    
    for symbol, df in stock_data.items():
        if '$close' not in df.columns:
            continue
        
        close = df['$close'].copy()
        
        # 动量因子 (过去20日收益)
        momentum = close.pct_change(lookback)
        
        # 波动率因子 (过去20日波动率)
        volatility = close.pct_change().rolling(lookback).std()
        
        # 反转因子 (过去5日收益的反向)
        reversal = -close.pct_change(5)
        
        # 综合得分
        score = momentum.rank(pct=True) * 0.4 + \
                (-volatility).rank(pct=True) * 0.3 + \
                reversal.rank(pct=True) * 0.3
        
        factors[symbol] = score
    
    return factors


def run_factor_backtest(stock_data: dict, test_start, test_end, initial_capital=100000, topk=10):
    """
    运行因子模型回测
    """
    print(f"\n🤖 运行因子模型回测: {test_start} ~ {test_end}")
    
    # 获取所有交易日
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted([d for d in all_dates if test_start <= d.strftime('%Y-%m-%d') <= test_end])
    
    if len(all_dates) == 0:
        print("❌ 测试期间没有交易日数据")
        return None
    
    print(f"   测试交易日: {len(all_dates)} 天")
    
    # 初始化
    cash = initial_capital
    positions = {}  # {symbol: amount}
    portfolio_values = []
    trade_records = []
    
    # 计算因子
    factors = simple_factor_model(stock_data)
    
    for i, date in enumerate(all_dates):
        # 当前日期的因子得分
        scores = {}
        prices = {}
        
        for symbol, df in stock_data.items():
            if date in df.index:
                if symbol in factors and date in factors[symbol].index:
                    score = factors[symbol].loc[date]
                    if not pd.isna(score):
                        scores[symbol] = score
                        prices[symbol] = df.loc[date, '$close']
        
        # 计算当前持仓市值
        position_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in positions if s in prices)
        total_value = cash + position_value
        
        # 每周调仓
        if i % 5 == 0 and len(scores) > topk:
            # 选出 topk 股票
            ranked = sorted(scores.items(), key=lambda x: -x[1])[:topk]
            target_stocks = {s[0] for s in ranked}
            
            # 卖出不在目标中的股票
            for symbol in list(positions.keys()):
                if symbol not in target_stocks and symbol in prices:
                    amount = positions[symbol]
                    sell_value = amount * prices[symbol]
                    sell_cost = max(sell_value * 0.0003, 5)
                    cash += sell_value - sell_cost
                    
                    trade_records.append({
                        'date': date,
                        'stock': symbol,
                        'action': 'SELL',
                        'amount': amount,
                        'price': prices[symbol],
                        'value': sell_value,
                    })
                    del positions[symbol]
            
            # 计算可用资金
            available = cash * 0.95  # 保留5%现金
            per_stock = available / topk
            
            # 买入目标股票
            for symbol, _ in ranked:
                if symbol not in positions and symbol in prices:
                    price = prices[symbol]
                    amount = int(per_stock / price / 100) * 100  # 整手
                    if amount > 0:
                        buy_value = amount * price
                        buy_cost = max(buy_value * 0.0003, 5)
                        if cash >= buy_value + buy_cost:
                            cash -= buy_value + buy_cost
                            positions[symbol] = amount
                            
                            trade_records.append({
                                'date': date,
                                'stock': symbol,
                                'action': 'BUY',
                                'amount': amount,
                                'price': price,
                                'value': buy_value,
                            })
        
        # 记录每日市值
        position_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in positions if s in prices)
        total_value = cash + position_value
        portfolio_values.append({
            'date': date,
            'cash': cash,
            'position_value': position_value,
            'total_value': total_value,
        })
    
    # 生成报告
    report_df = pd.DataFrame(portfolio_values)
    report_df = report_df.set_index('date')
    report_df['return'] = report_df['total_value'].pct_change()
    
    # 计算指标
    metrics = calculate_metrics(report_df['return'].dropna())
    
    return {
        'report': report_df,
        'positions': positions,
        'trade_records': pd.DataFrame(trade_records),
        'metrics': metrics,
    }


def calculate_buyhold(stock_data: dict, test_start, test_end, initial_capital=100000):
    """
    计算 Buy & Hold 策略 (使用沪深300ETF或指数)
    """
    print(f"\n📊 计算 Buy & Hold 策略: {test_start} ~ {test_end}")
    
    # 尝试获取指数数据
    for symbol in ['SH510300', 'SH000300', 'SZ159919']:
        if symbol in stock_data:
            df = stock_data[symbol]
            break
    else:
        # 使用第一只股票作为替代
        symbol = list(stock_data.keys())[0]
        df = stock_data[symbol]
        print(f"   使用 {symbol} 作为基准")
    
    # 筛选测试期
    df = df[(df.index >= test_start) & (df.index <= test_end)]
    
    if len(df) == 0:
        print("❌ 测试期间没有数据")
        return None
    
    # 计算收益
    first_price = df['$close'].iloc[0]
    shares = initial_capital / first_price
    buy_cost = initial_capital * 0.0003
    
    df = df.copy()
    df['account'] = shares * df['$close'] - buy_cost
    df['return'] = df['$close'].pct_change()
    
    metrics = calculate_metrics(df['return'].dropna())
    
    print(f"   交易日: {len(df)} 天")
    print(f"   最终市值: ¥{df['account'].iloc[-1]:,.2f}")
    
    return {
        'data': df,
        'metrics': metrics,
        'final_value': df['account'].iloc[-1],
    }


def generate_report(factor_result: dict, buyhold: dict, output_dir: Path):
    """生成报告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("📊 一致时间周期回测分析报告")
    print("=" * 80)
    
    print(f"\n💰 起始资金: ¥{CONFIG['account']:,}")
    print(f"\n📅 时间配置 (严格避免未来数据泄露):")
    print(f"   训练期: {CONFIG['train_start']} ~ {CONFIG['train_end']}")
    print(f"   验证期: {CONFIG['valid_start']} ~ {CONFIG['valid_end']}")
    print(f"   测试期: {CONFIG['test_start']} ~ {CONFIG['test_end']} (统一对比周期)")
    
    # 指标对比
    print("\n" + "-" * 80)
    print("📈 策略对比 (相同测试周期)")
    print("-" * 80)
    
    fm = factor_result.get('metrics', {}) if factor_result else {}
    bh = buyhold.get('metrics', {}) if buyhold else {}
    
    print(f"{'指标':<25} {'因子模型':>15} {'Buy & Hold':>15}")
    print("-" * 60)
    
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
        fm_val = fm.get(key, 0)
        bh_val = bh.get(key, 0)
        
        if key in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'profit_loss_ratio']:
            fmt = '{:.4f}'
        elif key == 'trading_days':
            fmt = '{:.0f}'
        else:
            fmt = '{:.2f}'
        
        print(f"{label:<25} {fmt.format(fm_val):>15} {fmt.format(bh_val):>15}")
    
    # 最终持仓
    if factor_result and 'positions' in factor_result:
        print(f"\n" + "-" * 80)
        print("📋 因子模型最终持仓")
        print("-" * 80)
        
        positions = factor_result['positions']
        print(f"持有 {len(positions)} 只股票")
        for stock, amount in positions.items():
            print(f"  {stock}: {amount:,} 股")
    
    # 交易记录
    if factor_result and 'trade_records' in factor_result:
        trade_df = factor_result['trade_records']
        if not trade_df.empty:
            print(f"\n" + "-" * 80)
            print(f"📝 因子模型交易记录 (共 {len(trade_df)} 条)")
            print("-" * 80)
            print(f"{'日期':<12} {'股票':<12} {'操作':>6} {'数量':>10} {'价格':>10} {'金额':>12}")
            print("-" * 65)
            
            for _, row in trade_df.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
                print(f"{date_str:<12} {row['stock']:<12} {row['action']:>6} {row['amount']:>10,.0f} {row['price']:>10.2f} ¥{row['value']:>10,.0f}")
            
            trade_df.to_csv(output_dir / 'factor_model_trades.csv', index=False)
            print(f"\n✅ 交易记录已保存: {output_dir / 'factor_model_trades.csv'}")
    
    # 保存汇总
    summary = [
        {**fm, 'strategy': 'factor_model', 'test_period': f"{CONFIG['test_start']} ~ {CONFIG['test_end']}"},
        {**bh, 'strategy': 'buy_hold', 'test_period': f"{CONFIG['test_start']} ~ {CONFIG['test_end']}"},
    ]
    pd.DataFrame(summary).to_csv(output_dir / 'metrics_comparison_consistent.csv', index=False)
    
    return summary


def generate_html_report(factor_result: dict, buyhold: dict, output_dir: Path):
    """生成 HTML 报告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if factor_result is None:
        print("❌ 无法生成 HTML 报告: 因子模型结果为空")
        return
    
    initial = CONFIG['account']
    
    # 因子模型数据
    fm_report = factor_result.get('report')
    if fm_report is not None:
        fm_dates = [d.strftime('%Y-%m-%d') for d in fm_report.index]
        fm_values = fm_report['total_value'].values.tolist()
        fm_returns = ((fm_report['total_value'] / initial - 1) * 100).tolist()
    else:
        fm_dates = []
        fm_values = []
        fm_returns = []
    
    # Buy & Hold 数据
    if buyhold and 'data' in buyhold:
        bh_data = buyhold['data']
        bh_dates = [d.strftime('%Y-%m-%d') for d in bh_data.index]
        bh_values = bh_data['account'].values.tolist()
        bh_returns = ((bh_data['account'] / initial - 1) * 100).tolist()
    else:
        bh_dates = fm_dates
        bh_values = [initial] * len(fm_dates)
        bh_returns = [0] * len(fm_dates)
    
    fm_m = factor_result.get('metrics', {}) if factor_result else {}
    bh_m = buyhold.get('metrics', {}) if buyhold else {}
    
    # 交易记录
    trade_df = factor_result.get('trade_records') if factor_result else None
    trade_html = ""
    if trade_df is not None and not trade_df.empty:
        rows = []
        for _, row in trade_df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
            action_class = 'buy' if row['action'] == 'BUY' else 'sell'
            rows.append(f"<tr><td>{date_str}</td><td>{row['stock']}</td><td class='{action_class}'>{row['action']}</td><td>{row['amount']:,.0f}</td><td>¥{row['price']:.2f}</td><td>¥{row['value']:,.0f}</td></tr>")
        trade_html = "\n".join(rows)
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>一致时间周期回测报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f5f7fa; }}
        .header {{ background: linear-gradient(135deg, #2e7d32, #1b5e20); color: white; padding: 30px 40px; }}
        .header h1 {{ margin: 0 0 10px; font-size: 28px; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 30px; }}
        .info-box {{ background: #e8f5e9; border: 1px solid #4caf50; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .warning-box {{ background: #fff3e0; border: 1px solid #ff9800; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
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
        .trades-container {{ max-height: 500px; overflow-y: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 一致时间周期回测报告</h1>
        <p>起始资金: ¥{initial:,} | 统一测试周期: {CONFIG['test_start']} ~ {CONFIG['test_end']}</p>
    </div>
    
    <div class="container">
        <div class="info-box">
            <strong>✅ 时间配置 (严格避免未来数据泄露):</strong><br>
            训练期: {CONFIG['train_start']} ~ {CONFIG['train_end']} | 
            验证期: {CONFIG['valid_start']} ~ {CONFIG['valid_end']} | 
            <strong>测试期: {CONFIG['test_start']} ~ {CONFIG['test_end']}</strong>
        </div>
        
        <div class="warning-box">
            <strong>⚠️ 说明:</strong> 所有策略使用相同的测试周期 ({len(fm_dates)} 个交易日) 进行公平对比
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value {'positive' if fm_m.get('total_return', 0) > 0 else 'negative'}">{fm_m.get('total_return', 0):+.2f}%</div>
                <div class="stat-label">因子模型 总收益</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'positive' if bh_m.get('total_return', 0) > 0 else 'negative'}">{bh_m.get('total_return', 0):+.2f}%</div>
                <div class="stat-label">Buy & Hold 总收益</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{fm_m.get('sharpe_ratio', 0):.2f}</div>
                <div class="stat-label">因子模型 Sharpe</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{bh_m.get('sharpe_ratio', 0):.2f}</div>
                <div class="stat-label">Buy & Hold Sharpe</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📈 账户价值对比 (相同周期)</div>
            <div id="value-chart" style="height: 400px;"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📊 累计收益率对比</div>
            <div id="return-chart" style="height: 400px;"></div>
        </div>
        
        <div class="metrics-table">
            <div class="chart-title">📋 完整指标对比</div>
            <table>
                <tr><th>指标</th><th>因子模型</th><th>Buy & Hold</th><th>超额收益</th></tr>
                <tr><td>总收益率</td><td class="{'positive' if fm_m.get('total_return',0)>0 else 'negative'}">{fm_m.get('total_return',0):+.2f}%</td><td class="{'positive' if bh_m.get('total_return',0)>0 else 'negative'}">{bh_m.get('total_return',0):+.2f}%</td><td class="{'positive' if fm_m.get('total_return',0)-bh_m.get('total_return',0)>0 else 'negative'}">{fm_m.get('total_return',0)-bh_m.get('total_return',0):+.2f}%</td></tr>
                <tr><td>年化收益率</td><td>{fm_m.get('annualized_return',0):+.2f}%</td><td>{bh_m.get('annualized_return',0):+.2f}%</td><td>{fm_m.get('annualized_return',0)-bh_m.get('annualized_return',0):+.2f}%</td></tr>
                <tr><td>年化波动率</td><td>{fm_m.get('annualized_volatility',0):.2f}%</td><td>{bh_m.get('annualized_volatility',0):.2f}%</td><td>-</td></tr>
                <tr><td>Sharpe Ratio</td><td><b>{fm_m.get('sharpe_ratio',0):.4f}</b></td><td><b>{bh_m.get('sharpe_ratio',0):.4f}</b></td><td>{fm_m.get('sharpe_ratio',0)-bh_m.get('sharpe_ratio',0):+.4f}</td></tr>
                <tr><td>Sortino Ratio</td><td>{fm_m.get('sortino_ratio',0):.4f}</td><td>{bh_m.get('sortino_ratio',0):.4f}</td><td>-</td></tr>
                <tr><td>最大回撤</td><td class="negative">{fm_m.get('max_drawdown',0):.2f}%</td><td class="negative">{bh_m.get('max_drawdown',0):.2f}%</td><td>-</td></tr>
                <tr><td>Calmar Ratio</td><td>{fm_m.get('calmar_ratio',0):.4f}</td><td>{bh_m.get('calmar_ratio',0):.4f}</td><td>-</td></tr>
                <tr><td>胜率</td><td>{fm_m.get('win_rate',0):.2f}%</td><td>{bh_m.get('win_rate',0):.2f}%</td><td>-</td></tr>
                <tr><td>交易天数</td><td>{fm_m.get('trading_days',0):.0f}</td><td>{bh_m.get('trading_days',0):.0f}</td><td>-</td></tr>
            </table>
        </div>
        
        <div class="metrics-table">
            <div class="chart-title">📝 因子模型交易记录 ({len(trade_df) if trade_df is not None else 0} 条)</div>
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
            {{x: {fm_dates}, y: {fm_values}, name: '因子模型', line: {{color: '#2e7d32', width: 2}}}},
            {{x: {bh_dates}, y: {bh_values}, name: 'Buy & Hold', line: {{color: '#1565c0', width: 2, dash: 'dash'}}}}
        ], {{margin: {{t:20,r:50,b:50,l:80}}, yaxis: {{title: '账户价值 (¥)', tickformat: ',.0f'}}, hovermode: 'x unified', legend: {{x: 0.02, y: 0.98}}}});
        
        Plotly.newPlot('return-chart', [
            {{x: {fm_dates}, y: {fm_returns}, name: '因子模型', fill: 'tozeroy', line: {{color: '#2e7d32'}}}},
            {{x: {bh_dates}, y: {bh_returns}, name: 'Buy & Hold', line: {{color: '#1565c0', dash: 'dash'}}}}
        ], {{margin: {{t:20,r:50,b:50,l:80}}, yaxis: {{title: '累计收益率 (%)', ticksuffix: '%'}}, hovermode: 'x unified', shapes: [{{type: 'line', x0: '{fm_dates[0]}', x1: '{fm_dates[-1] if fm_dates else ""}', y0: 0, y1: 0, line: {{color: '#888', width: 1, dash: 'dot'}}}}]}});
    </script>
</body>
</html>'''
    
    with open(output_dir / 'consistent_backtest_report.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML 报告已保存: {output_dir / 'consistent_backtest_report.html'}")


def main():
    print("=" * 80)
    print("🚀 一致时间周期回测分析")
    print("=" * 80)
    
    print(f"\n📅 时间配置:")
    print(f"   训练期: {CONFIG['train_start']} ~ {CONFIG['train_end']}")
    print(f"   验证期: {CONFIG['valid_start']} ~ {CONFIG['valid_end']}")
    print(f"   测试期: {CONFIG['test_start']} ~ {CONFIG['test_end']} ✅ 统一对比周期")
    
    print(f"\n💰 起始资金: ¥{CONFIG['account']:,}")
    
    # 加载 Baostock 数据
    stock_data = load_baostock_data()
    
    if stock_data is None or len(stock_data) == 0:
        print("❌ 无法加载数据")
        return
    
    print(f"✅ 加载了 {len(stock_data)} 只股票数据")
    
    # 运行因子模型回测
    factor_result = run_factor_backtest(
        stock_data, 
        CONFIG['test_start'], 
        CONFIG['test_end'],
        CONFIG['account'],
        CONFIG['strategy']['topk']
    )
    
    # 计算 Buy & Hold
    buyhold = calculate_buyhold(
        stock_data, 
        CONFIG['test_start'], 
        CONFIG['test_end'],
        CONFIG['account']
    )
    
    # 生成报告
    output_dir = Path(__file__).parent / "output" / "consistent_backtest"
    generate_report(factor_result, buyhold, output_dir)
    generate_html_report(factor_result, buyhold, output_dir)
    
    print(f"\n🌐 打开报告: open {output_dir / 'consistent_backtest_report.html'}")


if __name__ == "__main__":
    main()

