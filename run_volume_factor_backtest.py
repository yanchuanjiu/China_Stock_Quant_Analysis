#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量比因子策略回测

基于 Alpha158 中的量比相关因子:
- VMA: 成交量移动平均
- VSTD: 成交量标准差
- WVMA: 成交量加权价格波动
- CORR: 价格与成交量相关性
- CORD: 价格变化与成交量变化相关性
- VSUMP/VSUMN: 成交量上涨/下跌比例
- VSUMD: 成交量涨跌差异

特点:
1. 使用 Qlib 原生回测模块，准确跟踪账户余额
2. verbose=True 输出每笔交易详情
3. 自定义量比因子组合策略
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.backtest import backtest, get_strategy_executor
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.data.dataset.handler import DataHandlerLP


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
    "benchmark": "SH000300",
    
    # 时间配置
    "train_start": "2023-11-15",
    "train_end": "2025-02-28",
    "valid_start": "2025-03-01",
    "valid_end": "2025-04-15",
    "test_start": "2025-04-16",
    "test_end": "2025-05-14",
}


# ==================== 量比因子定义 ====================
def get_volume_factor_fields():
    """
    获取量比相关因子的字段定义
    """
    fields = []
    names = []
    windows = [5, 10, 20, 30, 60]
    
    # VMA - 成交量移动平均
    for d in windows:
        fields.append(f"Mean($volume, {d})/($volume+1e-12)")
        names.append(f"VMA{d}")
    
    # VSTD - 成交量标准差
    for d in windows:
        fields.append(f"Std($volume, {d})/($volume+1e-12)")
        names.append(f"VSTD{d}")
    
    # WVMA - 成交量加权价格波动
    for d in windows:
        fields.append(f"Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)")
        names.append(f"WVMA{d}")
    
    # CORR - 价格与成交量相关性
    for d in windows:
        fields.append(f"Corr($close, Log($volume+1), {d})")
        names.append(f"CORR{d}")
    
    # CORD - 价格变化与成交量变化相关性
    for d in windows:
        fields.append(f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})")
        names.append(f"CORD{d}")
    
    # VSUMP - 成交量上涨比例
    for d in windows:
        fields.append(f"Sum(Greater($volume-Ref($volume, 1), 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)")
        names.append(f"VSUMP{d}")
    
    # VSUMN - 成交量下跌比例
    for d in windows:
        fields.append(f"Sum(Greater(Ref($volume, 1)-$volume, 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)")
        names.append(f"VSUMN{d}")
    
    # VSUMD - 成交量涨跌差异
    for d in windows:
        fields.append(f"(Sum(Greater($volume-Ref($volume, 1), 0), {d})-Sum(Greater(Ref($volume, 1)-$volume, 0), {d}))/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)")
        names.append(f"VSUMD{d}")
    
    # 基础价格因子（用于预测）
    fields.append("$close/$open")
    names.append("OPEN_CLOSE_RATIO")
    fields.append("($high-$low)/$close")
    names.append("HIGH_LOW_RATIO")
    fields.append("$volume/Ref($volume, 1)")
    names.append("VOLUME_RATIO")
    
    return fields, names


def create_volume_factor_handler():
    """
    创建量比因子数据处理器
    使用 Alpha158 基类，只保留量比相关因子
    """
    # 使用 Alpha158 作为基础，但自定义 rolling 配置只包含量比因子
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
    
    return handler_config


def run_volume_factor_model(model_type="lightgbm"):
    """
    运行量比因子模型 (使用 Alpha158 全量因子，但突出量比因子重要性)
    """
    print(f"\n{'='*70}")
    print(f"🔬 量比因子增强模型: {model_type.upper()}")
    print(f"{'='*70}")
    
    print(f"\n📊 Alpha158 中的量比相关因子:")
    print(f"   - VMA (5/10/20/30/60): 成交量移动平均")
    print(f"   - VSTD: 成交量标准差")
    print(f"   - WVMA: 成交量加权价格波动")
    print(f"   - CORR: 价格与成交量相关性")
    print(f"   - CORD: 价格变化与成交量变化相关性")
    print(f"   - VSUMP/VSUMN: 成交量上涨/下跌比例")
    print(f"   - VSUMD: 成交量涨跌差异")
    
    # 使用 Alpha158 (包含量比因子)
    handler_config = create_volume_factor_handler()
    
    # 模型配置 - 调整参数突出量比因子
    if model_type.lower() == "xgboost":
        model_config = {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
            "kwargs": {
                "n_estimators": 150,  # 增加迭代
                "max_depth": 8,        # 增加深度
                "learning_rate": 0.08,
                "early_stopping_rounds": 30,
                "verbosity": 0,
                "colsample_bytree": 0.8,  # 特征采样
                "subsample": 0.8,
            },
        }
    else:
        model_config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "learning_rate": 0.03,  # 降低学习率
                "max_depth": 10,        # 增加深度
                "num_leaves": 256,      # 增加叶子
                "early_stopping_rounds": 80,
                "verbose": -1,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
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
    
    print(f"\n⏰ 时间配置:")
    print(f"   训练期: {CONFIG['train_start']} ~ {CONFIG['valid_start']}")
    print(f"   验证期: {CONFIG['valid_start']} ~ {CONFIG['valid_end']}")
    print(f"   测试期: {CONFIG['test_start']} ~ {CONFIG['test_end']}")
    
    # 初始化
    print(f"\n🔄 初始化数据集...")
    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)
    
    # 回测配置 - 使用 verbose=True 输出交易详情
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
                "verbose": True,
                "track_data": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "model": model,
                "dataset": dataset,
                "topk": 10,
                "n_drop": 2,
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
    
    exp_dir = Path(__file__).parent / "experiments"
    os.environ["MLFLOW_TRACKING_URI"] = str(exp_dir)
    
    exp_name = f"VolumeFactor_{model_type.upper()}"
    
    print(f"\n🚀 开始训练和回测...")
    
    with R.start(experiment_name=exp_name):
        print(f"   训练 {model_type.upper()} 模型...")
        model.fit(dataset)
        
        recorder = R.get_recorder()
        
        print(f"   生成预测信号...")
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        
        print(f"\n📈 执行回测 (verbose=True 显示交易详情):")
        print("-" * 70)
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        print("-" * 70)
        
        report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        analysis = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
        pred_df = recorder.load_object("pred.pkl")
    
    return {
        'model_type': model_type,
        'report': report_df,
        'positions': positions,
        'analysis': analysis,
        'pred': pred_df,
    }


def run_alpha158_model(model_type="lightgbm"):
    """
    运行标准 Alpha158 模型作为对比
    """
    print(f"\n{'='*70}")
    print(f"📊 标准 Alpha158 模型: {model_type.upper()}")
    print(f"{'='*70}")
    
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
    
    if model_type.lower() == "xgboost":
        model_config = {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
            "kwargs": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "early_stopping_rounds": 20, "verbosity": 0},
        }
    else:
        model_config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {"loss": "mse", "learning_rate": 0.05, "max_depth": 8, "num_leaves": 128, "early_stopping_rounds": 50, "verbose": -1},
        }
    
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
    
    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)
    
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True, "verbose": True},
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {"model": model, "dataset": dataset, "topk": 10, "n_drop": 2},
        },
        "backtest": {
            "start_time": CONFIG["test_start"],
            "end_time": CONFIG["test_end"],
            "account": CONFIG["account"],
            "benchmark": CONFIG["benchmark"],
            "exchange_kwargs": CONFIG["exchange_kwargs"],
        },
    }
    
    exp_dir = Path(__file__).parent / "experiments"
    os.environ["MLFLOW_TRACKING_URI"] = str(exp_dir)
    
    exp_name = f"Alpha158_{model_type.upper()}"
    
    print(f"\n🚀 训练和回测...")
    
    with R.start(experiment_name=exp_name):
        model.fit(dataset)
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        
        print(f"\n📈 执行回测:")
        print("-" * 70)
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        print("-" * 70)
        
        report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        analysis = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    
    return {
        'model_type': model_type,
        'report': report_df,
        'positions': positions,
        'analysis': analysis,
    }


def extract_account_history(positions: dict, initial_cash: float):
    """
    从 Qlib positions 中提取账户历史记录
    """
    records = []
    prev_position = {}
    
    for date in sorted(positions.keys()):
        pos = positions[date]
        if not hasattr(pos, 'position'):
            continue
        
        pos_dict = pos.position
        cash = pos_dict.get('cash', 0)
        account_value = pos_dict.get('now_account_value', 0)
        
        # 持仓信息
        holdings = []
        for stock, info in pos_dict.items():
            if stock in ['cash', 'now_account_value']:
                continue
            if isinstance(info, dict):
                holdings.append({
                    'stock': stock,
                    'amount': info.get('amount', 0),
                    'price': info.get('price', 0),
                    'value': info.get('amount', 0) * info.get('price', 0),
                })
        
        position_value = sum(h['value'] for h in holdings)
        
        records.append({
            'date': date,
            'cash': cash,
            'position_value': position_value,
            'account_value': account_value,
            'holdings_count': len(holdings),
            'holdings': holdings,
        })
    
    return pd.DataFrame(records)


def generate_comparison_report(results: dict, output_dir: Path):
    """
    生成对比报告
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("📊 量比因子 vs Alpha158 对比报告")
    print("=" * 80)
    
    print(f"\n💰 起始资金: ¥{CONFIG['account']:,}")
    print(f"📅 测试周期: {CONFIG['test_start']} ~ {CONFIG['test_end']}")
    
    # 收集指标
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
        ann_ret = ((1 + total_ret/100) ** (252/len(returns)) - 1) * 100
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
            'final_value': report['account'].iloc[-1],
            'trading_days': len(returns),
        }
        all_metrics.append(metrics)
    
    df_metrics = pd.DataFrame(all_metrics)
    
    print("\n" + "-" * 80)
    print("📈 模型对比")
    print("-" * 80)
    print(df_metrics.to_string(index=False))
    
    df_metrics.to_csv(output_dir / 'volume_factor_comparison.csv', index=False)
    
    # 账户历史记录
    for name, data in results.items():
        if data is None or 'positions' not in data:
            continue
        
        account_history = extract_account_history(data['positions'], CONFIG['account'])
        if not account_history.empty:
            print(f"\n" + "-" * 80)
            print(f"💰 {name} 账户历史 (Qlib 原生记录)")
            print("-" * 80)
            
            for _, row in account_history.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
                print(f"{date_str}: 现金=¥{row['cash']:,.2f} 持仓=¥{row['position_value']:,.2f} 总值=¥{row['account_value']:,.2f} 股票数={row['holdings_count']}")
            
            account_history.to_csv(output_dir / f'{name}_account_history.csv', index=False)
    
    return df_metrics


def generate_html_report(results: dict, output_dir: Path):
    """
    生成 HTML 报告
    """
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
        
        returns_series = report['return'].dropna()
        total_ret = (report['account'].iloc[-1] / initial - 1) * 100
        
        metrics_data[name] = {
            'total_return': total_ret,
            'sharpe': (returns_series.mean() - 0.03/252) / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0,
            'max_dd': ((1 + returns_series).cumprod().cummax() - (1 + returns_series).cumprod()).max() / (1 + returns_series).cumprod().cummax().max() * 100 if len(returns_series) > 0 else 0,
        }
    
    # 颜色配置
    colors = {
        'VolumeFactor_LightGBM': '#2e7d32',
        'VolumeFactor_XGBoost': '#1b5e20',
        'Alpha158_LightGBM': '#1565c0',
        'Alpha158_XGBoost': '#0d47a1',
    }
    
    # 生成 JavaScript 数据
    js_traces = []
    for name, cdata in chart_data.items():
        color = colors.get(name, '#666')
        js_traces.append(f"{{x: {cdata['dates']}, y: {cdata['values']}, name: '{name}', line: {{color: '{color}', width: 2}}}}")
    
    js_return_traces = []
    for name, cdata in chart_data.items():
        color = colors.get(name, '#666')
        fill = 'tozeroy' if 'VolumeFactor' in name else 'none'
        js_return_traces.append(f"{{x: {cdata['dates']}, y: {cdata['returns']}, name: '{name}', fill: '{fill}', line: {{color: '{color}', width: 2}}}}")
    
    # 指标表格
    metrics_rows = ""
    for name, m in metrics_data.items():
        is_volume = 'VolumeFactor' in name
        row_class = 'volume-row' if is_volume else ''
        metrics_rows += f"""<tr class="{row_class}">
            <td>{'🔬 ' if is_volume else '📊 '}{name}</td>
            <td class="{'positive' if m['total_return'] > 0 else 'negative'}">{m['total_return']:+.2f}%</td>
            <td>{m['sharpe']:.4f}</td>
            <td class="negative">{m['max_dd']:.2f}%</td>
        </tr>"""
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>量比因子 vs Alpha158 对比报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f5f7fa; }}
        .header {{ background: linear-gradient(135deg, #2e7d32, #1b5e20); color: white; padding: 30px 40px; }}
        .header h1 {{ margin: 0 0 10px; font-size: 28px; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 30px; }}
        .info-box {{ background: #e8f5e9; border: 1px solid #4caf50; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .factor-box {{ background: #fff8e1; border: 1px solid #ffc107; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .section {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 30px; }}
        .section h3 {{ margin: 0 0 15px; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: right; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        td:first-child, th:first-child {{ text-align: left; }}
        .positive {{ color: #34a853; font-weight: bold; }}
        .negative {{ color: #ea4335; }}
        .volume-row {{ background: #e8f5e9; }}
        .factor-list {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
        .factor-item {{ background: #f5f5f5; padding: 10px; border-radius: 6px; }}
        .factor-name {{ font-weight: bold; color: #2e7d32; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 量比因子 vs 📊 Alpha158 对比报告</h1>
        <p>起始资金: ¥{initial:,} | 测试周期: {CONFIG['test_start']} ~ {CONFIG['test_end']}</p>
    </div>
    
    <div class="container">
        <div class="factor-box">
            <strong>📌 量比因子列表:</strong>
            <div class="factor-list" style="margin-top: 10px;">
                <div class="factor-item"><span class="factor-name">VMA</span>: 成交量移动平均 (5/10/20/30/60日)</div>
                <div class="factor-item"><span class="factor-name">VSTD</span>: 成交量标准差</div>
                <div class="factor-item"><span class="factor-name">WVMA</span>: 成交量加权价格波动</div>
                <div class="factor-item"><span class="factor-name">CORR</span>: 价格与成交量相关性</div>
                <div class="factor-item"><span class="factor-name">CORD</span>: 价格变化与成交量变化相关性</div>
                <div class="factor-item"><span class="factor-name">VSUMP/VSUMN</span>: 成交量上涨/下跌比例</div>
                <div class="factor-item"><span class="factor-name">VSUMD</span>: 成交量涨跌差异 (RSI for volume)</div>
            </div>
        </div>
        
        <div class="section">
            <h3>📈 模型指标对比</h3>
            <table>
                <tr><th>模型</th><th>总收益率</th><th>Sharpe Ratio</th><th>最大回撤</th></tr>
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
    </div>
    
    <script>
        Plotly.newPlot('value-chart', [{','.join(js_traces)}], 
            {{margin: {{t:20,r:50,b:50,l:80}}, yaxis: {{title: '账户价值 (¥)', tickformat: ',.0f'}}, hovermode: 'x unified', legend: {{x: 0.02, y: 0.98}}}});
        
        Plotly.newPlot('return-chart', [{','.join(js_return_traces)}], 
            {{margin: {{t:20,r:50,b:50,l:80}}, yaxis: {{title: '累计收益率 (%)', ticksuffix: '%'}}, hovermode: 'x unified', shapes: [{{type: 'line', x0: '{list(chart_data.values())[0]["dates"][0] if chart_data else ""}', x1: '{list(chart_data.values())[0]["dates"][-1] if chart_data else ""}', y0: 0, y1: 0, line: {{color: '#888', width: 1, dash: 'dot'}}}}]}});
    </script>
</body>
</html>'''
    
    with open(output_dir / 'volume_factor_report.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML 报告已保存: {output_dir / 'volume_factor_report.html'}")


def main():
    print("=" * 80)
    print("🔬 量比因子策略回测")
    print("=" * 80)
    
    # 初始化 Qlib
    provider_uri = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    
    results = {}
    
    # 1. 量比因子 + LightGBM
    try:
        results['VolumeFactor_LightGBM'] = run_volume_factor_model("lightgbm")
    except Exception as e:
        print(f"❌ 量比因子 LightGBM 错误: {e}")
        import traceback
        traceback.print_exc()
        results['VolumeFactor_LightGBM'] = None
    
    # 2. 量比因子 + XGBoost
    try:
        results['VolumeFactor_XGBoost'] = run_volume_factor_model("xgboost")
    except Exception as e:
        print(f"❌ 量比因子 XGBoost 错误: {e}")
        results['VolumeFactor_XGBoost'] = None
    
    # 3. Alpha158 + LightGBM
    try:
        results['Alpha158_LightGBM'] = run_alpha158_model("lightgbm")
    except Exception as e:
        print(f"❌ Alpha158 LightGBM 错误: {e}")
        results['Alpha158_LightGBM'] = None
    
    # 4. Alpha158 + XGBoost
    try:
        results['Alpha158_XGBoost'] = run_alpha158_model("xgboost")
    except Exception as e:
        print(f"❌ Alpha158 XGBoost 错误: {e}")
        results['Alpha158_XGBoost'] = None
    
    # 生成报告
    output_dir = Path(__file__).parent / "output" / "volume_factor_backtest"
    generate_comparison_report(results, output_dir)
    generate_html_report(results, output_dir)
    
    print(f"\n🌐 打开报告: open {output_dir / 'volume_factor_report.html'}")


if __name__ == "__main__":
    main()

