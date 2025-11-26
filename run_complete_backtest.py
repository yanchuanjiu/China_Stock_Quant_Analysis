#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整回测脚本 - 包含模型策略 vs Buy & Hold 对比

配置:
- 起始资金: 10万元
- 交易单位: 100股/手
- 手续费: 万分之三 (买卖各万3)
- 滑点: 0.1%
- 成交价: 开盘价 (信号后第二天开盘)
- 回测周期: 使用数据可用的最大范围
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import pandas as pd
import numpy as np

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord


# ==================== 实盘配置 ====================
CONFIG = {
    # 账户配置 - 10万起始资金
    "account": 100000,
    
    # 交易所配置
    "exchange_kwargs": {
        "freq": "day",
        "limit_threshold": 0.095,      # 涨跌停限制 9.5%
        "deal_price": "open",          # 使用开盘价成交
        "open_cost": 0.0003,           # 买入手续费 万3
        "close_cost": 0.0003,          # 卖出手续费 万3
        "min_cost": 5,                 # 最低手续费 5元
        "impact_cost": 0.001,          # 滑点 0.1%
        "trade_unit": 100,             # 100股/手
    },
    
    # 策略配置
    "strategy": {
        "topk": 10,      # 持有股票数量
        "n_drop": 2,     # 每天换仓数量
    },
    
    "benchmark": "SH000300",  # 沪深300指数
}


def get_data_range():
    """获取数据可用范围"""
    # 检查实际数据截止日期
    data = D.features(['SH600000'], ['$close'], start_time='2024-01-01', end_time='2025-12-31', freq='day')
    data = data.dropna()
    
    if len(data) == 0:
        raise ValueError("没有可用的股票数据")
    
    last_date = data.index[-1][1]
    return last_date


def calculate_buy_and_hold(start_date, end_date, initial_capital=100000):
    """
    计算 Buy & Hold 策略收益
    使用沪深300ETF (SH510300) 或沪深300指数模拟
    """
    print(f"\n📊 计算 Buy & Hold 策略 ({start_date} ~ {end_date})...")
    
    # 尝试获取沪深300ETF或指数数据
    etf_codes = ['SH510300', 'SH000300']  # 300ETF 和 沪深300指数
    
    for code in etf_codes:
        try:
            data = D.features([code], ['$close', '$open'], 
                            start_time=start_date, end_time=end_date, freq='day')
            data = data.dropna()
            
            if len(data) > 10:
                print(f"  使用 {code} 数据 ({len(data)} 天)")
                break
        except:
            continue
    else:
        print("  ⚠️ 无法获取ETF/指数数据，使用基准收益率")
        return None
    
    # 计算每日收益
    data = data.reset_index()
    data.columns = ['instrument', 'datetime', 'close', 'open']
    data = data.sort_values('datetime')
    
    # 使用开盘价买入（与策略一致）
    first_open = data['open'].iloc[0]
    shares = initial_capital / first_open
    
    # 计算每日账户价值
    data['account_value'] = shares * data['close']
    data['daily_return'] = data['close'].pct_change()
    data['cumulative_return'] = (data['close'] / first_open - 1) * 100
    
    # 计算手续费（买入一次）
    buy_cost = initial_capital * 0.0003
    data['account_value'] = data['account_value'] - buy_cost
    
    result = {
        'datetime': data['datetime'].values,
        'account_value': data['account_value'].values,
        'daily_return': data['daily_return'].values,
        'cumulative_return': data['cumulative_return'].values,
        'final_value': data['account_value'].iloc[-1],
        'total_return': (data['account_value'].iloc[-1] / initial_capital - 1) * 100,
        'trading_days': len(data),
        'shares': shares,
        'etf_code': code,
    }
    
    print(f"  买入价格: ¥{first_open:.4f}")
    print(f"  买入份额: {shares:.2f}")
    print(f"  最终市值: ¥{result['final_value']:,.2f}")
    print(f"  总收益率: {result['total_return']:+.2f}%")
    
    return result


def run_model_backtest(start_date, end_date, model_type="lightgbm"):
    """运行模型回测"""
    print(f"\n🤖 运行 {model_type.upper()} 模型回测...")
    
    # 计算训练/验证期
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    train_end = (start_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    train_start = (start_dt - timedelta(days=365)).strftime('%Y-%m-%d')
    valid_start = (start_dt - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"  训练期: {train_start} ~ {valid_start}")
    print(f"  验证期: {valid_start} ~ {train_end}")
    print(f"  回测期: {start_date} ~ {end_date}")
    
    # 数据处理配置
    data_handler_config = {
        "start_time": train_start,
        "end_time": end_date,
        "fit_start_time": train_start,
        "fit_end_time": train_end,
        "instruments": "csi300",
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
            },
        }
        exp_name = "Backtest_XGBoost_10W"
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
            },
        }
        exp_name = "Backtest_LightGBM_10W"
    
    # 数据集配置
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
    
    # 创建模型和数据集
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
                "topk": CONFIG["strategy"]["topk"],
                "n_drop": CONFIG["strategy"]["n_drop"],
            },
        },
        "backtest": {
            "start_time": start_date,
            "end_time": end_date,
            "account": CONFIG["account"],
            "benchmark": CONFIG["benchmark"],
            "exchange_kwargs": CONFIG["exchange_kwargs"],
        },
    }
    
    # 设置实验目录
    exp_dir = Path(__file__).parent / "experiments"
    exp_dir.mkdir(exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = str(exp_dir)
    
    # 开始实验
    with R.start(experiment_name=exp_name):
        # 训练
        model.fit(dataset)
        R.save_objects(trained_model=model)
        
        # 预测
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        
        # 回测
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        
        # 获取结果
        report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        analysis = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
        
        exp_id = R.get_exp().id
        run_id = recorder.id
    
    return {
        'model_type': model_type,
        'exp_name': exp_name,
        'exp_id': exp_id,
        'run_id': run_id,
        'report': report_df,
        'positions': positions,
        'analysis': analysis,
        'start_date': start_date,
        'end_date': end_date,
    }


def generate_comparison_report(model_results, buyhold_result, output_dir):
    """生成对比报告"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_df = model_results['report']
    
    # 计算模型策略指标
    initial_capital = CONFIG["account"]
    final_value = report_df['account'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    total_cost = report_df['cost'].sum() * 100
    avg_turnover = report_df['turnover'].mean() * 100
    
    # 风险指标
    analysis = model_results['analysis']
    excess_with_cost = analysis.get('excess_return_with_cost', {})
    
    print("\n" + "=" * 70)
    print("📊 回测结果对比")
    print("=" * 70)
    
    print(f"\n💰 起始资金: ¥{initial_capital:,}")
    print(f"📅 回测周期: {model_results['start_date']} ~ {model_results['end_date']}")
    print(f"📈 交易天数: {len(report_df)}")
    
    print("\n" + "-" * 70)
    print(f"{'指标':<25} {'模型策略':>20} {'Buy & Hold':>20}")
    print("-" * 70)
    
    bh_return = buyhold_result['total_return'] if buyhold_result else 0
    bh_final = buyhold_result['final_value'] if buyhold_result else initial_capital
    
    print(f"{'最终市值':<25} {'¥{:,.2f}'.format(final_value):>20} {'¥{:,.2f}'.format(bh_final):>20}")
    print(f"{'总收益率':<25} {'{:+.2f}%'.format(total_return):>20} {'{:+.2f}%'.format(bh_return):>20}")
    print(f"{'超额收益':<25} {'{:+.2f}%'.format(total_return - bh_return):>20} {'-':>20}")
    print(f"{'累计交易成本':<25} {'{:.2f}%'.format(total_cost):>20} {'0.03%':>20}")
    print(f"{'平均换手率':<25} {'{:.2f}%'.format(avg_turnover):>20} {'0%':>20}")
    
    if excess_with_cost:
        ann_return = excess_with_cost.get('annualized_return', 0) * 100
        info_ratio = excess_with_cost.get('information_ratio', 0)
        max_dd = excess_with_cost.get('max_drawdown', 0) * 100
        
        print(f"\n{'年化超额收益':<25} {'{:+.2f}%'.format(ann_return):>20}")
        print(f"{'信息比率':<25} {'{:.4f}'.format(info_ratio):>20}")
        print(f"{'最大回撤':<25} {'{:.2f}%'.format(max_dd):>20}")
    
    print("-" * 70)
    
    # 显示持仓详情
    positions = model_results['positions']
    dates = sorted(positions.keys())
    last_date = dates[-1]
    last_pos = positions[last_date]
    
    print(f"\n📋 最终持仓 ({last_date.strftime('%Y-%m-%d')})")
    if hasattr(last_pos, 'position'):
        pos_dict = last_pos.position
        cash = pos_dict.get('cash', 0)
        account_value = pos_dict.get('now_account_value', 0)
        
        stocks = [(k, v) for k, v in pos_dict.items() 
                 if k not in ['cash', 'now_account_value'] and isinstance(v, dict)]
        
        print(f"  现金余额: ¥{cash:,.2f}")
        print(f"  账户总值: ¥{account_value:,.2f}")
        print(f"  持仓股票: {len(stocks)} 只")
        
        print(f"\n  {'股票代码':<12} {'持仓数量':>10} {'市值':>12} {'权重':>8}")
        print("  " + "-" * 50)
        
        for stock, info in sorted(stocks, key=lambda x: -x[1].get('weight', 0))[:10]:
            amount = info.get('amount', 0)
            price = info.get('price', 0)
            weight = info.get('weight', 0) * 100
            market_val = amount * price
            print(f"  {stock:<12} {amount:>10,.0f} ¥{market_val:>10,.0f} {weight:>7.2f}%")
    
    # 保存每日对比数据
    comparison_df = pd.DataFrame({
        'date': report_df.index,
        'model_account': report_df['account'].values,
        'model_return': report_df['return'].values * 100,
        'model_cost': report_df['cost'].values * 100,
        'model_turnover': report_df['turnover'].values * 100,
        'benchmark_return': report_df['bench'].values * 100,
    })
    
    if buyhold_result:
        # 对齐日期
        bh_df = pd.DataFrame({
            'date': buyhold_result['datetime'],
            'buyhold_account': buyhold_result['account_value'],
        })
        bh_df['date'] = pd.to_datetime(bh_df['date'])
        comparison_df['date'] = pd.to_datetime(comparison_df['date'])
        comparison_df = comparison_df.merge(bh_df, on='date', how='left')
    
    comparison_df.to_csv(output_dir / 'daily_comparison.csv', index=False)
    
    # 保存汇总
    summary = {
        'model_type': model_results['model_type'],
        'start_date': model_results['start_date'],
        'end_date': model_results['end_date'],
        'initial_capital': initial_capital,
        'model_final_value': final_value,
        'model_total_return': total_return,
        'buyhold_final_value': bh_final,
        'buyhold_total_return': bh_return,
        'excess_return': total_return - bh_return,
        'total_cost': total_cost,
        'avg_turnover': avg_turnover,
    }
    
    if excess_with_cost:
        summary.update({
            'annualized_excess_return': excess_with_cost.get('annualized_return', 0) * 100,
            'information_ratio': excess_with_cost.get('information_ratio', 0),
            'max_drawdown': excess_with_cost.get('max_drawdown', 0) * 100,
        })
    
    pd.DataFrame([summary]).to_csv(output_dir / 'summary.csv', index=False)
    
    # 保存持仓
    with open(output_dir / 'positions.pkl', 'wb') as f:
        pickle.dump(positions, f)
    
    print(f"\n✅ 结果已保存到: {output_dir}")
    
    return summary, comparison_df


def generate_html_report(model_results, buyhold_result, output_dir):
    """生成 HTML 可视化报告"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_df = model_results['report']
    initial_capital = CONFIG["account"]
    
    # 准备数据
    dates = [d.strftime('%Y-%m-%d') for d in report_df.index]
    model_values = report_df['account'].values.tolist()
    model_returns = ((report_df['account'] / initial_capital - 1) * 100).tolist()
    bench_returns = (report_df['bench'].cumsum() * 100).tolist()
    
    if buyhold_result:
        bh_dates = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in buyhold_result['datetime']]
        bh_values = buyhold_result['account_value'].tolist()
        bh_returns = buyhold_result['cumulative_return'].tolist()
    else:
        bh_dates = dates
        bh_values = [initial_capital] * len(dates)
        bh_returns = [0] * len(dates)
    
    # 计算统计
    final_model = model_values[-1]
    final_bh = bh_values[-1] if buyhold_result else initial_capital
    model_return = (final_model / initial_capital - 1) * 100
    bh_return = buyhold_result['total_return'] if buyhold_result else 0
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>回测对比报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; background: #f5f7fa; }}
        .header {{ background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%); color: white; padding: 30px 40px; }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 28px; }}
        .header p {{ margin: 0; opacity: 0.9; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 30px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); text-align: center; }}
        .stat-value {{ font-size: 32px; font-weight: bold; margin-bottom: 5px; }}
        .stat-label {{ color: #666; font-size: 14px; }}
        .positive {{ color: #34a853; }}
        .negative {{ color: #ea4335; }}
        .chart-container {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 30px; }}
        .chart-title {{ font-size: 18px; font-weight: 600; margin-bottom: 15px; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: right; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        td:first-child, th:first-child {{ text-align: left; }}
        .compare-table {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 回测对比报告</h1>
        <p>回测周期: {model_results['start_date']} ~ {model_results['end_date']} | 起始资金: ¥{initial_capital:,}</p>
    </div>
    
    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value {'positive' if model_return > 0 else 'negative'}">{model_return:+.2f}%</div>
                <div class="stat-label">{model_results['model_type'].upper()} 策略收益</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'positive' if bh_return > 0 else 'negative'}">{bh_return:+.2f}%</div>
                <div class="stat-label">Buy & Hold 收益</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'positive' if model_return - bh_return > 0 else 'negative'}">{model_return - bh_return:+.2f}%</div>
                <div class="stat-label">超额收益</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">¥{final_model:,.0f}</div>
                <div class="stat-label">策略最终市值</div>
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
        
        <div class="compare-table">
            <div class="chart-title">📋 策略对比明细</div>
            <table>
                <tr>
                    <th>指标</th>
                    <th>{model_results['model_type'].upper()} 策略</th>
                    <th>Buy & Hold</th>
                    <th>差异</th>
                </tr>
                <tr>
                    <td>起始资金</td>
                    <td>¥{initial_capital:,}</td>
                    <td>¥{initial_capital:,}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>最终市值</td>
                    <td>¥{final_model:,.2f}</td>
                    <td>¥{final_bh:,.2f}</td>
                    <td class="{'positive' if final_model > final_bh else 'negative'}">¥{final_model - final_bh:+,.2f}</td>
                </tr>
                <tr>
                    <td>总收益率</td>
                    <td class="{'positive' if model_return > 0 else 'negative'}">{model_return:+.2f}%</td>
                    <td class="{'positive' if bh_return > 0 else 'negative'}">{bh_return:+.2f}%</td>
                    <td class="{'positive' if model_return > bh_return else 'negative'}">{model_return - bh_return:+.2f}%</td>
                </tr>
                <tr>
                    <td>交易成本</td>
                    <td>{report_df['cost'].sum() * 100:.2f}%</td>
                    <td>0.03%</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>平均换手率</td>
                    <td>{report_df['turnover'].mean() * 100:.2f}%</td>
                    <td>0%</td>
                    <td>-</td>
                </tr>
            </table>
        </div>
    </div>
    
    <script>
        // 账户价值图
        var valueTrace1 = {{
            x: {dates},
            y: {model_values},
            name: '{model_results['model_type'].upper()} 策略',
            type: 'scatter',
            line: {{ color: '#1a73e8', width: 2 }}
        }};
        
        var valueTrace2 = {{
            x: {bh_dates},
            y: {bh_values},
            name: 'Buy & Hold',
            type: 'scatter',
            line: {{ color: '#ea4335', width: 2, dash: 'dash' }}
        }};
        
        Plotly.newPlot('value-chart', [valueTrace1, valueTrace2], {{
            margin: {{ t: 20, r: 50, b: 50, l: 80 }},
            xaxis: {{ title: '日期' }},
            yaxis: {{ title: '账户价值 (¥)', tickformat: ',.0f' }},
            legend: {{ x: 0.02, y: 0.98 }},
            hovermode: 'x unified'
        }});
        
        // 累计收益率图
        var returnTrace1 = {{
            x: {dates},
            y: {model_returns},
            name: '{model_results['model_type'].upper()} 策略',
            type: 'scatter',
            fill: 'tozeroy',
            line: {{ color: '#1a73e8', width: 2 }}
        }};
        
        var returnTrace2 = {{
            x: {bh_dates},
            y: {bh_returns},
            name: 'Buy & Hold',
            type: 'scatter',
            line: {{ color: '#ea4335', width: 2, dash: 'dash' }}
        }};
        
        Plotly.newPlot('return-chart', [returnTrace1, returnTrace2], {{
            margin: {{ t: 20, r: 50, b: 50, l: 80 }},
            xaxis: {{ title: '日期' }},
            yaxis: {{ title: '累计收益率 (%)', ticksuffix: '%' }},
            legend: {{ x: 0.02, y: 0.98 }},
            hovermode: 'x unified'
        }});
    </script>
</body>
</html>'''
    
    with open(output_dir / 'comparison_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML 报告已保存: {output_dir / 'comparison_report.html'}")


def main():
    print("=" * 70)
    print("🚀 完整回测分析 - 模型策略 vs Buy & Hold")
    print("=" * 70)
    
    # 初始化 Qlib
    provider_uri = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    print(f"✅ Qlib 初始化成功")
    
    # 获取数据范围
    last_date = get_data_range()
    print(f"📊 数据可用截止日期: {last_date.strftime('%Y-%m-%d')}")
    
    # 设置回测日期
    end_date = last_date.strftime('%Y-%m-%d')
    start_date = '2024-11-15'  # 从这个日期开始回测
    
    print(f"\n💰 回测配置:")
    print(f"   起始资金: ¥{CONFIG['account']:,}")
    print(f"   回测周期: {start_date} ~ {end_date}")
    print(f"   交易单位: {CONFIG['exchange_kwargs']['trade_unit']} 股/手")
    print(f"   手续费: 万{CONFIG['exchange_kwargs']['open_cost']*10000:.0f}")
    print(f"   滑点: {CONFIG['exchange_kwargs']['impact_cost']*100:.1f}%")
    
    # 计算 Buy & Hold 收益
    buyhold_result = calculate_buy_and_hold(start_date, end_date, CONFIG["account"])
    
    # 运行模型回测
    model_results = run_model_backtest(start_date, end_date, "lightgbm")
    
    # 生成报告
    output_dir = Path(__file__).parent / "output" / "backtest_comparison"
    summary, comparison_df = generate_comparison_report(model_results, buyhold_result, output_dir)
    generate_html_report(model_results, buyhold_result, output_dir)
    
    print(f"\n🌐 打开报告:")
    print(f"   open {output_dir / 'comparison_report.html'}")


if __name__ == "__main__":
    main()

