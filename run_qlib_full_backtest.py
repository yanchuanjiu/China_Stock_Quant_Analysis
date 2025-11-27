#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qlib 完整回测分析

解决的问题:
1. 每日交易（第二天开盘价执行）
2. 交易记录包含账户现金余额，确保没有超买
3. 多模型对比 (LightGBM + XGBoost)
4. 使用 Qlib 完整的回测指标和报告模块
5. 基于 Alpha158 量比因子的配置和回测

配置:
- 起始资金: 10万元
- 交易单位: 100股/手
- 手续费: 万分之三
- 滑点: 0.1%
- 成交价: 第二天开盘价
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report import analysis_model, analysis_position


# ==================== 配置 ====================
CONFIG = {
    "account": 100000,  # 起始资金 10万
    "exchange_kwargs": {
        "freq": "day",
        "limit_threshold": 0.095,  # 涨跌停限制
        "deal_price": "open",      # 第二天开盘价交易
        "open_cost": 0.0003,       # 买入手续费 万3
        "close_cost": 0.0003,      # 卖出手续费 万3
        "min_cost": 5,             # 最小手续费
        "impact_cost": 0.001,      # 滑点 0.1%
        "trade_unit": 100,         # 100股/手
    },
    "strategy": {
        "topk": 10,    # 持有前10只股票
        "n_drop": 2,   # 每次调仓最多卖出2只
    },
    "benchmark": "SH000300",
    
    # 时间配置 (避免过拟合)
    "train_start": "2023-11-15",
    "train_end": "2025-02-28",
    "valid_start": "2025-03-01",
    "valid_end": "2025-04-15",
    "test_start": "2025-04-16",
    "test_end": "2025-05-14",
}


# ==================== 量比相关因子配置 ====================
VOLUME_FACTOR_CONFIG = {
    "class": "Alpha158",
    "module_path": "qlib.contrib.data.handler",
    "kwargs": {
        "start_time": CONFIG["train_start"],
        "end_time": CONFIG["test_end"],
        "fit_start_time": CONFIG["train_start"],
        "fit_end_time": CONFIG["train_end"],
        "instruments": "csi300",
        # 只选择量比相关的因子
        "config": {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"],
            },
            "volume": {
                "windows": [0, 1, 2, 3, 4],  # 近5日成交量
            },
            "rolling": {
                "windows": [5, 10, 20, 30, 60],
                "include": [
                    "VMA",    # 成交量移动平均
                    "VSTD",   # 成交量标准差
                    "WVMA",   # 成交量加权价格波动
                    "CORR",   # 价格与成交量相关性
                    "CORD",   # 价格变化与成交量变化相关性
                    "VSUMP",  # 成交量上涨比例
                    "VSUMN",  # 成交量下跌比例
                    "VSUMD",  # 成交量涨跌差
                    "ROC",    # 价格变化率
                    "MA",     # 价格移动平均
                    "STD",    # 价格标准差
                ],
            },
        },
    },
}


def extract_detailed_trades(positions: dict, report_df: pd.DataFrame, initial_cash: float):
    """
    提取详细交易记录，包含账户余额
    """
    trade_records = []
    dates = sorted(positions.keys())
    
    prev_holdings = {}
    prev_cash = initial_cash
    
    for i, date in enumerate(dates):
        pos = positions[date]
        if not hasattr(pos, 'position'):
            continue
        
        pos_dict = pos.position
        current_cash = pos_dict.get('cash', prev_cash)
        account_value = pos_dict.get('now_account_value', current_cash)
        
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
        
        daily_trades = []
        for stock in all_stocks:
            prev_amt = prev_holdings.get(stock, {}).get('amount', 0)
            curr_amt = current_holdings.get(stock, {}).get('amount', 0)
            price = current_holdings.get(stock, {}).get('price', 0) or prev_holdings.get(stock, {}).get('price', 0)
            
            if abs(curr_amt - prev_amt) > 0.01:
                if curr_amt > prev_amt:
                    action = 'BUY'
                    amount = curr_amt - prev_amt
                    cost = amount * price * 0.0003 + amount * price * 0.001  # 手续费+滑点
                else:
                    action = 'SELL'
                    amount = prev_amt - curr_amt
                    cost = amount * price * 0.0003 + amount * price * 0.001
                
                daily_trades.append({
                    'date': date,
                    'stock': stock,
                    'action': action,
                    'amount': amount,
                    'price': price,
                    'value': amount * price,
                    'cost': cost,
                    'cash_before': prev_cash,
                    'cash_after': current_cash,
                    'account_value': account_value,
                })
        
        # 检查是否超买
        for trade in daily_trades:
            if trade['action'] == 'BUY':
                if trade['cash_before'] < trade['value'] + trade['cost']:
                    trade['warning'] = '⚠️ 现金不足'
                else:
                    trade['warning'] = ''
        
        trade_records.extend(daily_trades)
        prev_holdings = current_holdings
        prev_cash = current_cash
    
    return pd.DataFrame(trade_records)


def run_model_backtest(model_type="lightgbm", use_volume_factors=False):
    """
    运行单个模型回测
    """
    print(f"\n{'='*60}")
    print(f"🤖 训练 {model_type.upper()} 模型" + (" (量比因子)" if use_volume_factors else " (Alpha158)"))
    print(f"{'='*60}")
    
    # 数据处理器配置 - 使用标准 Alpha158
    handler_config = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": CONFIG["train_start"],
            "end_time": CONFIG["test_end"],
            "fit_start_time": CONFIG["train_start"],
            "fit_end_time": CONFIG["train_end"],
            "instruments": "csi300",
        },
    }
    
    # 模型配置
    if model_type.lower() == "xgboost":
        model_config = {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
            "kwargs": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "early_stopping_rounds": 20,
                "verbosity": 0,
            },
        }
    else:
        model_config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "learning_rate": 0.05,
                "max_depth": 8,
                "num_leaves": 128,
                "early_stopping_rounds": 50,
                "verbose": -1,
            },
        }
    
    # 数据集配置
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": (CONFIG["train_start"], CONFIG["valid_start"]),
                "valid": (CONFIG["valid_start"], CONFIG["valid_end"]),
                "test": (CONFIG["test_start"], CONFIG["test_end"]),
            },
        },
    }
    
    print(f"  训练期: {CONFIG['train_start']} ~ {CONFIG['valid_start']}")
    print(f"  验证期: {CONFIG['valid_start']} ~ {CONFIG['valid_end']}")
    print(f"  测试期: {CONFIG['test_start']} ~ {CONFIG['test_end']}")
    
    # 初始化
    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)
    
    # 回测配置
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "model": model,
                "dataset": dataset,
                **CONFIG["strategy"],
            },
        },
        "backtest": {
            "start_time": CONFIG["test_start"],
            "end_time": CONFIG["test_end"],
            "account": CONFIG["account"],
            "benchmark": CONFIG["benchmark"],
            "exchange_kwargs": CONFIG["exchange_kwargs"],
        },
    }
    
    # 实验目录
    exp_dir = Path(__file__).parent / "experiments"
    os.environ["MLFLOW_TRACKING_URI"] = str(exp_dir)
    
    suffix = "_VolumeFactor" if use_volume_factors else ""
    exp_name = f"FullBacktest_{model_type.upper()}{suffix}"
    
    with R.start(experiment_name=exp_name):
        # 训练模型
        print("  训练模型...")
        model.fit(dataset)
        
        recorder = R.get_recorder()
        
        # 生成信号
        print("  生成预测信号...")
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        
        # 回测分析
        print("  执行回测...")
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        
        # 加载结果
        report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        analysis = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
        pred_df = recorder.load_object("pred.pkl")
    
    # 提取交易记录
    trade_records = extract_detailed_trades(positions, report_df, CONFIG["account"])
    
    # Qlib 风险分析
    print("\n📊 Qlib 风险分析:")
    if 'excess_return_with_cost' in analysis:
        print(analysis['excess_return_with_cost'])
    else:
        # 计算基本指标
        returns = report_df['return'].dropna()
        total_ret = (report_df['account'].iloc[-1] / CONFIG['account'] - 1)
        print(f"  总收益率: {total_ret*100:.2f}%")
        print(f"  交易天数: {len(returns)}")
    
    return {
        'model_type': model_type,
        'use_volume_factors': use_volume_factors,
        'report': report_df,
        'positions': positions,
        'analysis': analysis,
        'trade_records': trade_records,
        'pred': pred_df,
    }


def calculate_buyhold(start_date, end_date, initial_capital=100000):
    """计算 Buy & Hold 策略"""
    print(f"\n📊 计算 Buy & Hold 策略...")
    
    for code in ['SH510300', 'SH000300']:
        try:
            data = D.features([code], ['$close', '$open'], 
                             start_time=start_date, end_time=end_date, freq='day')
            data = data.dropna()
            if len(data) > 5:
                print(f"  使用基准: {code}")
                break
        except:
            continue
    else:
        print("  ⚠️ 无法获取基准数据")
        return None
    
    data = data.reset_index()
    data.columns = ['instrument', 'datetime', 'close', 'open']
    data = data.sort_values('datetime')
    
    first_open = data['open'].iloc[0]
    shares = initial_capital / first_open
    buy_cost = initial_capital * 0.0003
    
    data['account'] = shares * data['close'] - buy_cost
    data['return'] = data['close'].pct_change()
    
    # 计算指标
    returns = data['return'].dropna()
    total_return = (data['account'].iloc[-1] / initial_capital - 1) * 100
    ann_return = (1 + total_return/100) ** (252/len(returns)) - 1
    ann_volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() - 0.03/252) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    max_dd = ((cumulative - cumulative.cummax()) / cumulative.cummax()).min()
    
    print(f"  交易日: {len(data)} 天")
    print(f"  总收益: {total_return:+.2f}%")
    
    return {
        'data': data,
        'metrics': {
            'total_return': total_return,
            'annualized_return': ann_return * 100,
            'annualized_volatility': ann_volatility * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd * 100,
            'trading_days': len(data),
        },
    }


def generate_comprehensive_report(results: dict, buyhold: dict, output_dir: Path):
    """生成完整报告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("📊 完整回测分析报告 (Qlib 框架)")
    print("=" * 80)
    
    print(f"\n💰 起始资金: ¥{CONFIG['account']:,}")
    print(f"📅 测试周期: {CONFIG['test_start']} ~ {CONFIG['test_end']}")
    print(f"⚙️ 交易配置: 开盘价成交, 万3手续费, 0.1%滑点, 100股/手")
    
    # 收集所有模型指标
    all_metrics = []
    for name, data in results.items():
        if data is None:
            continue
        
        report = data.get('report')
        if report is None:
            continue
            
        returns = report['return'].dropna()
        if len(returns) == 0:
            continue
            
        total_ret = (report['account'].iloc[-1] / CONFIG['account'] - 1) * 100
        ann_ret = ((1 + total_ret/100) ** (252/len(returns)) - 1) * 100 if len(returns) > 0 else 0
        ann_vol = returns.std() * np.sqrt(252) * 100
        sharpe = (returns.mean() - 0.03/252) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        max_dd = ((cumulative - cumulative.cummax()) / cumulative.cummax()).min() * 100
        
        metrics = {
            'model': name,
            'total_return': total_ret,
            'annualized_return': ann_ret,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'trading_days': len(returns),
        }
        
        all_metrics.append(metrics)
    
    if buyhold:
        all_metrics.append({
            'model': 'Buy & Hold',
            **buyhold['metrics'],
        })
    
    # 打印对比表
    print("\n" + "-" * 80)
    print("📈 模型对比")
    print("-" * 80)
    
    df_metrics = pd.DataFrame(all_metrics)
    print(df_metrics.to_string(index=False))
    df_metrics.to_csv(output_dir / 'metrics_comparison.csv', index=False)
    
    # 打印交易记录
    for name, data in results.items():
        if data is None:
            continue
        
        trade_df = data.get('trade_records')
        if trade_df is None or trade_df.empty:
            continue
        
        print(f"\n" + "-" * 80)
        print(f"📝 {name} 交易记录 (共 {len(trade_df)} 条)")
        print("-" * 80)
        print(f"{'日期':<12} {'股票':<12} {'操作':>6} {'数量':>8} {'价格':>10} {'金额':>12} {'现金(前)':>12} {'现金(后)':>12} {'账户总值':>12}")
        print("-" * 100)
        
        for _, row in trade_df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
            warning = row.get('warning', '')
            print(f"{date_str:<12} {row['stock']:<12} {row['action']:>6} {row['amount']:>8,.0f} {row['price']:>10.2f} ¥{row['value']:>10,.0f} ¥{row['cash_before']:>10,.0f} ¥{row['cash_after']:>10,.0f} ¥{row['account_value']:>10,.0f} {warning}")
        
        # 检查是否有超买
        if 'warning' in trade_df.columns:
            warnings = trade_df[trade_df['warning'].str.len() > 0]
            if len(warnings) > 0:
                print(f"\n⚠️ 警告: 发现 {len(warnings)} 条可能超买的记录")
            else:
                print(f"\n✅ 所有交易记录正常，无超买情况")
        
        trade_df.to_csv(output_dir / f'{name}_trades_detailed.csv', index=False)
        print(f"✅ 已保存: {output_dir / f'{name}_trades_detailed.csv'}")
    
    return all_metrics


def generate_html_report(results: dict, buyhold: dict, output_dir: Path):
    """生成 HTML 报告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    initial = CONFIG['account']
    
    # 收集数据
    chart_data = {}
    metrics_data = {}
    
    for name, data in results.items():
        if data is None or 'report' not in data:
            continue
        
        report = data['report']
        dates = [d.strftime('%Y-%m-%d') for d in report.index]
        values = report['account'].values.tolist()
        returns = ((report['account'] / initial - 1) * 100).tolist()
        
        chart_data[name] = {'dates': dates, 'values': values, 'returns': returns}
        
        analysis = data.get('analysis', {})
        excess = analysis.get('excess_return_with_cost', pd.DataFrame())
        if isinstance(excess, pd.DataFrame) and 'risk' in excess.columns:
            metrics_data[name] = {
                'ann_return': float(excess.loc['annualized_return', 'risk']) * 100,
                'max_dd': float(excess.loc['max_drawdown', 'risk']) * 100,
                'ir': float(excess.loc['information_ratio', 'risk']),
            }
    
    if buyhold and 'data' in buyhold:
        bh_data = buyhold['data']
        bh_dates = [d.strftime('%Y-%m-%d') for d in bh_data['datetime']]
        bh_values = bh_data['account'].values.tolist()
        bh_returns = ((bh_data['account'] / initial - 1) * 100).tolist()
        chart_data['Buy_Hold'] = {'dates': bh_dates, 'values': bh_values, 'returns': bh_returns}
        metrics_data['Buy_Hold'] = buyhold['metrics']
    
    # 交易记录 HTML
    trade_html = ""
    for name, data in results.items():
        if data is None:
            continue
        trade_df = data.get('trade_records')
        if trade_df is None or trade_df.empty:
            continue
        
        rows = []
        for _, row in trade_df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
            action_class = 'buy' if row['action'] == 'BUY' else 'sell'
            warning_class = 'warning' if row.get('warning', '') else ''
            rows.append(f"""<tr class="{warning_class}">
                <td>{date_str}</td>
                <td>{row['stock']}</td>
                <td class="{action_class}">{row['action']}</td>
                <td>{row['amount']:,.0f}</td>
                <td>¥{row['price']:.2f}</td>
                <td>¥{row['value']:,.0f}</td>
                <td>¥{row['cash_before']:,.0f}</td>
                <td>¥{row['cash_after']:,.0f}</td>
                <td>¥{row['account_value']:,.0f}</td>
            </tr>""")
        
        trade_html += f"""
        <div class="section">
            <h3>📝 {name} 交易明细 ({len(trade_df)} 条)</h3>
            <div class="trades-container">
            <table>
                <tr><th>日期</th><th>股票</th><th>操作</th><th>数量</th><th>价格</th><th>金额</th><th>现金(前)</th><th>现金(后)</th><th>账户总值</th></tr>
                {''.join(rows)}
            </table>
            </div>
        </div>
        """
    
    # 生成 JavaScript 数据
    js_traces = []
    colors = {'LightGBM': '#1a73e8', 'XGBoost': '#fbbc04', 'LightGBM_VolumeFactor': '#34a853', 'XGBoost_VolumeFactor': '#ea4335', 'Buy_Hold': '#9e9e9e'}
    
    for name, cdata in chart_data.items():
        color = colors.get(name, '#666')
        dash = 'dash' if 'Hold' in name else 'solid'
        js_traces.append(f"{{x: {cdata['dates']}, y: {cdata['values']}, name: '{name}', line: {{color: '{color}', width: 2, dash: '{dash}'}}}}")
    
    js_return_traces = []
    for name, cdata in chart_data.items():
        color = colors.get(name, '#666')
        dash = 'dash' if 'Hold' in name else 'solid'
        js_return_traces.append(f"{{x: {cdata['dates']}, y: {cdata['returns']}, name: '{name}', line: {{color: '{color}', width: 2, dash: '{dash}'}}}}")
    
    # 指标表格
    metrics_rows = ""
    for name, m in metrics_data.items():
        if isinstance(m, dict):
            metrics_rows += f"""<tr>
                <td>{name}</td>
                <td class="{'positive' if m.get('ann_return', m.get('annualized_return', 0)) > 0 else 'negative'}">{m.get('ann_return', m.get('annualized_return', 0)):+.2f}%</td>
                <td class="negative">{m.get('max_dd', m.get('max_drawdown', 0)):.2f}%</td>
                <td>{m.get('ir', m.get('sharpe_ratio', 0)):.4f}</td>
            </tr>"""
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Qlib 完整回测报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f5f7fa; }}
        .header {{ background: linear-gradient(135deg, #0d47a1, #1565c0); color: white; padding: 30px 40px; }}
        .header h1 {{ margin: 0 0 10px; font-size: 28px; }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 30px; }}
        .config-box {{ background: #e3f2fd; border: 1px solid #2196f3; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .section {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 30px; }}
        .section h3 {{ margin: 0 0 15px; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        td:first-child, th:first-child {{ text-align: left; }}
        .buy {{ color: #ea4335; font-weight: bold; }}
        .sell {{ color: #34a853; font-weight: bold; }}
        .positive {{ color: #34a853; }}
        .negative {{ color: #ea4335; }}
        .warning {{ background: #fff3e0; }}
        .trades-container {{ max-height: 500px; overflow-y: auto; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; }}
        .stat-label {{ font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Qlib 完整回测报告</h1>
        <p>起始资金: ¥{initial:,} | 测试周期: {CONFIG['test_start']} ~ {CONFIG['test_end']}</p>
    </div>
    
    <div class="container">
        <div class="config-box">
            <strong>⚙️ 交易配置:</strong>
            成交价=第二天开盘价 | 买入手续费=万3 | 卖出手续费=万3 | 滑点=0.1% | 交易单位=100股/手 | 涨跌停限制=9.5%
        </div>
        
        <div class="section">
            <h3>📈 模型指标对比</h3>
            <table>
                <tr><th>模型</th><th>年化收益</th><th>最大回撤</th><th>信息比率/Sharpe</th></tr>
                {metrics_rows}
            </table>
        </div>
        
        <div class="section">
            <h3>💰 账户价值走势</h3>
            <div id="value-chart" style="height: 400px;"></div>
        </div>
        
        <div class="section">
            <h3>📊 累计收益率对比</h3>
            <div id="return-chart" style="height: 400px;"></div>
        </div>
        
        {trade_html}
    </div>
    
    <script>
        Plotly.newPlot('value-chart', [{','.join(js_traces)}], 
            {{margin: {{t:20,r:50,b:50,l:80}}, yaxis: {{title: '账户价值 (¥)', tickformat: ',.0f'}}, hovermode: 'x unified', legend: {{x: 0.02, y: 0.98}}}});
        
        Plotly.newPlot('return-chart', [{','.join(js_return_traces)}], 
            {{margin: {{t:20,r:50,b:50,l:80}}, yaxis: {{title: '累计收益率 (%)', ticksuffix: '%'}}, hovermode: 'x unified', shapes: [{{type: 'line', x0: '{list(chart_data.values())[0]["dates"][0] if chart_data else ""}', x1: '{list(chart_data.values())[0]["dates"][-1] if chart_data else ""}', y0: 0, y1: 0, line: {{color: '#888', width: 1, dash: 'dot'}}}}]}});
    </script>
</body>
</html>'''
    
    with open(output_dir / 'qlib_full_report.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML 报告已保存: {output_dir / 'qlib_full_report.html'}")


def main():
    print("=" * 80)
    print("🚀 Qlib 完整回测分析")
    print("=" * 80)
    
    # 初始化 Qlib
    provider_uri = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    
    results = {}
    
    # 1. LightGBM (标准 Alpha158)
    try:
        results['LightGBM'] = run_model_backtest("lightgbm", use_volume_factors=False)
    except Exception as e:
        print(f"❌ LightGBM 错误: {e}")
        results['LightGBM'] = None
    
    # 2. XGBoost (标准 Alpha158)
    try:
        results['XGBoost'] = run_model_backtest("xgboost", use_volume_factors=False)
    except Exception as e:
        print(f"❌ XGBoost 错误: {e}")
        results['XGBoost'] = None
    
    # 量比因子模型暂时跳过（Alpha158 自定义配置需要特殊处理）
    
    # 5. Buy & Hold
    buyhold = calculate_buyhold(CONFIG['test_start'], CONFIG['test_end'], CONFIG['account'])
    
    # 生成报告
    output_dir = Path(__file__).parent / "output" / "qlib_full_backtest"
    generate_comprehensive_report(results, buyhold, output_dir)
    generate_html_report(results, buyhold, output_dir)
    
    print(f"\n🌐 打开报告: open {output_dir / 'qlib_full_report.html'}")


if __name__ == "__main__":
    main()

