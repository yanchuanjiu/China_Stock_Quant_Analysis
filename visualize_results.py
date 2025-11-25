#!/usr/bin/env python3
"""
Qlib 实验结果可视化脚本
============================
功能：
1. 收益曲线图 (Cumulative Return)
2. 风险分析图 (Risk Analysis)
3. IC 分析图 (Score IC)
4. 模型性能对比图
5. 持仓详情
"""

import os
import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np

# 检查并安装可视化依赖
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    print("安装 plotly...")
    os.system("pip install plotly kaleido")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px


def load_experiment_data(exp_id: str, exp_dir: Path):
    """加载实验数据"""
    exp_path = exp_dir / exp_id
    data = {}
    
    # 查找 run 目录
    for run_dir in exp_path.iterdir():
        if run_dir.is_dir() and run_dir.name not in ['meta.yaml', '.trash']:
            artifacts_dir = run_dir / 'artifacts'
            
            # 加载预测结果
            pred_file = artifacts_dir / 'pred.pkl'
            if pred_file.exists():
                with open(pred_file, 'rb') as f:
                    data['pred'] = pickle.load(f)
            
            # 加载标签
            label_file = artifacts_dir / 'label.pkl'
            if label_file.exists():
                with open(label_file, 'rb') as f:
                    data['label'] = pickle.load(f)
            
            # 加载回测报告
            report_file = artifacts_dir / 'portfolio_analysis' / 'report_normal_1day.pkl'
            if report_file.exists():
                with open(report_file, 'rb') as f:
                    data['report'] = pickle.load(f)
            
            # 加载持仓
            positions_file = artifacts_dir / 'portfolio_analysis' / 'positions_normal_1day.pkl'
            if positions_file.exists():
                with open(positions_file, 'rb') as f:
                    data['positions'] = pickle.load(f)
            
            # 加载分析结果
            analysis_file = artifacts_dir / 'portfolio_analysis' / 'port_analysis_1day.pkl'
            if analysis_file.exists():
                with open(analysis_file, 'rb') as f:
                    data['analysis'] = pickle.load(f)
            
            # 加载 IC 分析
            ic_file = artifacts_dir / 'sig_analysis' / 'ic.pkl'
            if ic_file.exists():
                with open(ic_file, 'rb') as f:
                    data['ic'] = pickle.load(f)
            
            # 加载 metrics
            metrics_dir = run_dir / 'metrics'
            if metrics_dir.exists():
                data['metrics'] = {}
                for metric_file in metrics_dir.iterdir():
                    with open(metric_file) as f:
                        content = f.read().strip().split()
                        if len(content) >= 2:
                            data['metrics'][metric_file.name] = float(content[1])
            
            break
    
    return data


def create_cumulative_return_chart(data_dict: dict, output_dir: Path):
    """创建累计收益曲线图"""
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('累计收益曲线对比', '每日收益分布'),
                        vertical_spacing=0.15,
                        row_heights=[0.7, 0.3])
    
    colors = {'XGBoost': '#1f77b4', 'LightGBM': '#ff7f0e', 'Benchmark': '#2ca02c'}
    
    for model_name, data in data_dict.items():
        if 'report' not in data:
            continue
        
        report_df = data['report']
        if report_df is None or report_df.empty:
            continue
        
        # 计算累计收益
        if 'return' in report_df.columns:
            cum_return = (1 + report_df['return']).cumprod() - 1
            
            fig.add_trace(
                go.Scatter(x=cum_return.index, y=cum_return.values * 100,
                          name=f'{model_name} 策略收益',
                          line=dict(color=colors.get(model_name, '#1f77b4'), width=2)),
                row=1, col=1
            )
            
            # 每日收益柱状图
            fig.add_trace(
                go.Bar(x=report_df.index, y=report_df['return'].values * 100,
                      name=f'{model_name} 每日收益',
                      marker_color=colors.get(model_name, '#1f77b4'),
                      opacity=0.6),
                row=2, col=1
            )
        
        # 基准收益
        if 'bench' in report_df.columns and model_name == list(data_dict.keys())[0]:
            cum_bench = (1 + report_df['bench']).cumprod() - 1
            fig.add_trace(
                go.Scatter(x=cum_bench.index, y=cum_bench.values * 100,
                          name='沪深300 基准',
                          line=dict(color=colors['Benchmark'], width=2, dash='dash')),
                row=1, col=1
            )
    
    fig.update_layout(
        title='📈 模型回测收益曲线对比',
        height=800,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified'
    )
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="累计收益率 (%)", row=1, col=1)
    fig.update_yaxes(title_text="每日收益率 (%)", row=2, col=1)
    
    # 保存
    output_file = output_dir / 'cumulative_return.html'
    fig.write_html(str(output_file))
    print(f"✅ 累计收益图已保存: {output_file}")
    
    return fig


def create_risk_analysis_chart(data_dict: dict, output_dir: Path):
    """创建风险分析图"""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('最大回撤', '月度收益热力图', '收益分布', '滚动夏普比率'),
                        vertical_spacing=0.12,
                        horizontal_spacing=0.1)
    
    colors = {'XGBoost': '#1f77b4', 'LightGBM': '#ff7f0e'}
    
    for model_name, data in data_dict.items():
        if 'report' not in data:
            continue
        
        report_df = data['report']
        if report_df is None or report_df.empty:
            continue
        
        if 'return' in report_df.columns:
            returns = report_df['return']
            cum_return = (1 + returns).cumprod()
            
            # 最大回撤
            rolling_max = cum_return.expanding().max()
            drawdown = (cum_return - rolling_max) / rolling_max * 100
            
            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown.values,
                          name=f'{model_name} 回撤',
                          fill='tozeroy',
                          line=dict(color=colors.get(model_name, '#1f77b4'))),
                row=1, col=1
            )
            
            # 收益分布直方图
            fig.add_trace(
                go.Histogram(x=returns.values * 100, 
                            name=f'{model_name} 收益分布',
                            marker_color=colors.get(model_name, '#1f77b4'),
                            opacity=0.7,
                            nbinsx=50),
                row=2, col=1
            )
            
            # 滚动夏普比率 (20日)
            rolling_sharpe = returns.rolling(20).mean() / returns.rolling(20).std() * np.sqrt(252)
            fig.add_trace(
                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                          name=f'{model_name} 滚动夏普',
                          line=dict(color=colors.get(model_name, '#1f77b4'))),
                row=2, col=2
            )
    
    fig.update_layout(
        title='📊 风险分析仪表板',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    fig.update_yaxes(title_text="回撤 (%)", row=1, col=1)
    fig.update_yaxes(title_text="频次", row=2, col=1)
    fig.update_xaxes(title_text="日收益率 (%)", row=2, col=1)
    fig.update_yaxes(title_text="夏普比率", row=2, col=2)
    
    output_file = output_dir / 'risk_analysis.html'
    fig.write_html(str(output_file))
    print(f"✅ 风险分析图已保存: {output_file}")
    
    return fig


def create_model_comparison_chart(data_dict: dict, output_dir: Path):
    """创建模型对比图"""
    metrics_data = []
    
    for model_name, data in data_dict.items():
        if 'metrics' not in data:
            continue
        
        metrics = data['metrics']
        row = {
            '模型': model_name,
            'IC': metrics.get('IC', 0),
            'ICIR': metrics.get('ICIR', 0),
            'Rank IC': metrics.get('Rank IC', 0),
            '年化收益(含成本)': metrics.get('1day.excess_return_with_cost.annualized_return', 0) * 100,
            '信息比率': metrics.get('1day.excess_return_with_cost.information_ratio', 0),
            '最大回撤': abs(metrics.get('1day.excess_return_with_cost.max_drawdown', 0)) * 100,
        }
        metrics_data.append(row)
    
    if not metrics_data:
        print("⚠️ 没有可用的指标数据")
        return None
    
    df = pd.DataFrame(metrics_data)
    
    # 创建雷达图
    categories = ['IC', 'ICIR', 'Rank IC', '年化收益(%)', '信息比率']
    
    fig = go.Figure()
    
    colors = {'XGBoost': '#1f77b4', 'LightGBM': '#ff7f0e'}
    
    for _, row in df.iterrows():
        model = row['模型']
        # 归一化数据用于雷达图
        values = [
            row['IC'] * 10,  # 放大IC
            row['ICIR'],
            row['Rank IC'] * 10,
            row['年化收益(含成本)'] / 10,
            row['信息比率'],
        ]
        values.append(values[0])  # 闭合
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=model,
            line_color=colors.get(model, '#1f77b4'),
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='🎯 模型性能雷达图对比',
        showlegend=True,
        height=600
    )
    
    output_file = output_dir / 'model_comparison_radar.html'
    fig.write_html(str(output_file))
    print(f"✅ 模型对比雷达图已保存: {output_file}")
    
    # 创建柱状图对比
    fig2 = make_subplots(rows=2, cols=3,
                         subplot_titles=('IC', 'ICIR', 'Rank IC', '年化收益(%)', '信息比率', '最大回撤(%)'))
    
    metrics_cols = ['IC', 'ICIR', 'Rank IC', '年化收益(含成本)', '信息比率', '最大回撤']
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
    
    for i, (col, pos) in enumerate(zip(metrics_cols, positions)):
        for j, row in df.iterrows():
            fig2.add_trace(
                go.Bar(x=[row['模型']], y=[row[col]], 
                       name=row['模型'] if i == 0 else None,
                       marker_color=colors.get(row['模型'], '#1f77b4'),
                       showlegend=(i == 0)),
                row=pos[0], col=pos[1]
            )
    
    fig2.update_layout(
        title='📊 模型指标详细对比',
        height=600,
        showlegend=True
    )
    
    output_file2 = output_dir / 'model_comparison_bars.html'
    fig2.write_html(str(output_file2))
    print(f"✅ 模型对比柱状图已保存: {output_file2}")
    
    return fig, fig2


def create_positions_table(data_dict: dict, output_dir: Path):
    """创建持仓详情表"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>交易持仓详情</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        h2 { color: #1a73e8; margin-top: 30px; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }
        h3 { color: #666; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; background: white; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: right; }
        th { background-color: #1a73e8; color: white; }
        td:first-child { text-align: left; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        tr:hover { background-color: #e8f0fe; }
        .positive { color: #34a853; font-weight: bold; }
        .negative { color: #ea4335; font-weight: bold; }
        .summary { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }
        .stat-card { background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stat-value { font-size: 24px; font-weight: bold; color: #1a73e8; }
        .stat-label { font-size: 12px; color: #666; margin-top: 5px; }
    </style>
</head>
<body>
    <h1>📋 交易持仓详情</h1>
    <p style="text-align: center; color: #666;">回测周期: 2024-10-08 ~ 2025-05-14 (146 个交易日)</p>
"""
    
    for model_name, data in data_dict.items():
        positions = data.get('positions')
        report_df = data.get('report')
        metrics = data.get('metrics', {})
        
        html_content += f"<h2>🤖 {model_name} 模型</h2>"
        
        # 统计卡片
        annual_return = metrics.get('1day.excess_return_with_cost.annualized_return', 0) * 100
        info_ratio = metrics.get('1day.excess_return_with_cost.information_ratio', 0)
        max_dd = metrics.get('1day.excess_return_with_cost.max_drawdown', 0) * 100
        
        html_content += f'''
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value {'positive' if annual_return > 0 else 'negative'}">{annual_return:+.2f}%</div>
                <div class="stat-label">年化超额收益</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{info_ratio:.4f}</div>
                <div class="stat-label">信息比率</div>
            </div>
            <div class="stat-card">
                <div class="stat-value negative">{max_dd:.2f}%</div>
                <div class="stat-label">最大回撤</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{metrics.get("IC", 0):.4f}</div>
                <div class="stat-label">IC</div>
            </div>
        </div>
        '''
        
        # 详细交易记录
        if report_df is not None and not report_df.empty:
            html_content += "<h3>📈 每日交易记录 (最近30天)</h3>"
            html_content += '''<table>
                <tr>
                    <th>日期</th>
                    <th>每日收益</th>
                    <th>基准收益</th>
                    <th>超额收益</th>
                    <th>换手率</th>
                    <th>交易成本</th>
                    <th>账户价值</th>
                    <th>现金</th>
                </tr>'''
            
            for date, row in report_df.tail(30).iterrows():
                ret = row.get('return', 0) * 100
                bench = row.get('bench', 0) * 100
                excess = ret - bench
                turnover = row.get('turnover', 0) * 100
                cost = row.get('cost', 0) * 100
                account = row.get('account', 0)
                cash = row.get('cash', 0)
                
                ret_class = 'positive' if ret > 0 else 'negative'
                excess_class = 'positive' if excess > 0 else 'negative'
                
                html_content += f'''<tr>
                    <td>{date.strftime('%Y-%m-%d')}</td>
                    <td class="{ret_class}">{ret:+.4f}%</td>
                    <td>{bench:+.4f}%</td>
                    <td class="{excess_class}">{excess:+.4f}%</td>
                    <td>{turnover:.2f}%</td>
                    <td>{cost:.4f}%</td>
                    <td>{account:,.0f}</td>
                    <td>{cash:,.0f}</td>
                </tr>'''
            
            html_content += "</table>"
        
        # 持仓详情
        if positions is not None:
            dates = sorted(positions.keys())
            last_date = dates[-1]
            last_pos = positions[last_date]
            
            if hasattr(last_pos, 'position'):
                pos_dict = last_pos.position
                stocks = [(k, v) for k, v in pos_dict.items() if k not in ['cash', 'now_account_value'] and isinstance(v, dict)]
                
                html_content += f"<h3>📊 最终持仓 ({last_date.strftime('%Y-%m-%d')})</h3>"
                html_content += f'''<div class="summary">
                    <p><strong>持仓股票数:</strong> {len(stocks)}</p>
                    <p><strong>现金余额:</strong> ¥{pos_dict.get("cash", 0):,.2f}</p>
                    <p><strong>账户总值:</strong> ¥{pos_dict.get("now_account_value", 0):,.2f}</p>
                </div>'''
                
                html_content += '''<table>
                    <tr>
                        <th>股票代码</th>
                        <th>持仓数量</th>
                        <th>最新价格</th>
                        <th>持仓市值</th>
                        <th>持仓权重</th>
                        <th>持有天数</th>
                    </tr>'''
                
                # 按权重排序
                stocks_sorted = sorted(stocks, key=lambda x: -x[1].get('weight', 0))
                for stock, info in stocks_sorted:
                    amount = info.get('amount', 0)
                    price = info.get('price', 0)
                    weight = info.get('weight', 0) * 100
                    days = info.get('count_day', 0)
                    market_value = amount * price
                    
                    html_content += f'''<tr>
                        <td>{stock}</td>
                        <td>{amount:,.0f}</td>
                        <td>¥{price:.4f}</td>
                        <td>¥{market_value:,.0f}</td>
                        <td>{weight:.2f}%</td>
                        <td>{days}</td>
                    </tr>'''
                
                html_content += "</table>"
    
    html_content += "</body></html>"
    
    output_file = output_dir / 'positions_detail.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ 持仓详情已保存: {output_file}")


def create_summary_dashboard(data_dict: dict, output_dir: Path):
    """创建汇总仪表板"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>RD-Agent 量化实验可视化仪表板</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: white; text-align: center; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        .card h3 { margin-top: 0; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }
        .metric-name { color: #666; }
        .metric-value { font-weight: bold; }
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        .link-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
        .link-card { background: white; border-radius: 8px; padding: 15px; text-align: center; text-decoration: none; color: #333; transition: transform 0.3s, box-shadow 0.3s; }
        .link-card:hover { transform: translateY(-5px); box-shadow: 0 15px 35px rgba(0,0,0,0.3); }
        .link-card .icon { font-size: 32px; margin-bottom: 10px; }
        .winner { background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 RD-Agent 量化实验可视化仪表板</h1>
        
        <div class="card-grid">
"""
    
    # 为每个模型创建卡片
    for model_name, data in data_dict.items():
        metrics = data.get('metrics', {})
        
        ic = metrics.get('IC', 0)
        icir = metrics.get('ICIR', 0)
        annual_return = metrics.get('1day.excess_return_with_cost.annualized_return', 0) * 100
        info_ratio = metrics.get('1day.excess_return_with_cost.information_ratio', 0)
        max_dd = metrics.get('1day.excess_return_with_cost.max_drawdown', 0) * 100
        
        ret_class = 'positive' if annual_return > 0 else 'negative'
        
        html_content += f"""
            <div class="card">
                <h3>🤖 {model_name}</h3>
                <div class="metric"><span class="metric-name">IC</span><span class="metric-value">{ic:.4f}</span></div>
                <div class="metric"><span class="metric-name">ICIR</span><span class="metric-value">{icir:.4f}</span></div>
                <div class="metric"><span class="metric-name">年化收益(含成本)</span><span class="metric-value {ret_class}">{annual_return:+.2f}%</span></div>
                <div class="metric"><span class="metric-name">信息比率</span><span class="metric-value">{info_ratio:.4f}</span></div>
                <div class="metric"><span class="metric-name">最大回撤</span><span class="metric-value negative">{max_dd:.2f}%</span></div>
            </div>
"""
    
    html_content += """
        </div>
        
        <h2 style="color: white; text-align: center;">📊 详细分析报告</h2>
        <div class="link-grid">
            <a href="cumulative_return.html" class="link-card">
                <div class="icon">📈</div>
                <div>累计收益曲线</div>
            </a>
            <a href="risk_analysis.html" class="link-card">
                <div class="icon">⚠️</div>
                <div>风险分析</div>
            </a>
            <a href="model_comparison_radar.html" class="link-card">
                <div class="icon">🎯</div>
                <div>模型雷达图</div>
            </a>
            <a href="model_comparison_bars.html" class="link-card">
                <div class="icon">📊</div>
                <div>指标对比</div>
            </a>
            <a href="positions_detail.html" class="link-card">
                <div class="icon">📋</div>
                <div>持仓详情</div>
            </a>
        </div>
    </div>
</body>
</html>
"""
    
    output_file = output_dir / 'index.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ 仪表板已保存: {output_file}")


def main():
    print("=" * 60)
    print("📊 Qlib 实验结果可视化")
    print("=" * 60)
    
    # 实验目录
    exp_dir = PROJECT_ROOT / 'experiments'
    output_dir = PROJECT_ROOT / 'output' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 实验 ID 映射 (使用正确数据范围 2024-10-08 ~ 2025-05-14 的实验)
    experiments = {
        'XGBoost': '339731341714338232',
        'LightGBM': '587758757846340364'
    }
    
    # 加载所有实验数据
    print("\n📂 加载实验数据...")
    data_dict = {}
    for model_name, exp_id in experiments.items():
        print(f"  加载 {model_name} (ID: {exp_id})...")
        data = load_experiment_data(exp_id, exp_dir)
        if data:
            data_dict[model_name] = data
            print(f"    ✓ 已加载: {list(data.keys())}")
        else:
            print(f"    ⚠️ 数据加载失败")
    
    if not data_dict:
        print("❌ 没有可用的实验数据")
        return
    
    # 生成图表
    print("\n📈 生成可视化图表...")
    
    try:
        create_cumulative_return_chart(data_dict, output_dir)
    except Exception as e:
        print(f"  ⚠️ 累计收益图生成失败: {e}")
    
    try:
        create_risk_analysis_chart(data_dict, output_dir)
    except Exception as e:
        print(f"  ⚠️ 风险分析图生成失败: {e}")
    
    try:
        create_model_comparison_chart(data_dict, output_dir)
    except Exception as e:
        print(f"  ⚠️ 模型对比图生成失败: {e}")
    
    try:
        create_positions_table(data_dict, output_dir)
    except Exception as e:
        print(f"  ⚠️ 持仓详情生成失败: {e}")
    
    try:
        create_summary_dashboard(data_dict, output_dir)
    except Exception as e:
        print(f"  ⚠️ 仪表板生成失败: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 可视化完成!")
    print("=" * 60)
    print(f"\n📁 结果位置: {output_dir}")
    print(f"\n🌐 打开浏览器查看:")
    print(f"   file://{output_dir / 'index.html'}")
    print(f"\n   或使用命令:")
    print(f"   open {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()

