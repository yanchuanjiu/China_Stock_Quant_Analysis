#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模拟实盘交易脚本

配置说明 (全部使用 Qlib 原生配置项):
- 起始资金: 10万元 (account: 100000)
- 交易单位: 100股/手 (trade_unit: 100, 中国市场默认)
- 手续费: 万分之三 (open_cost: 0.0003, close_cost: 0.0003)
- 滑点: 0.1% (impact_cost: 0.001)
- 成交价: 信号后第二天开盘价 (deal_price: "open")
- 涨跌停限制: 9.5% (limit_threshold: 0.095)
- 最低手续费: 5元 (min_cost: 5)

使用方法:
    python run_paper_trading.py              # 运行回测
    python run_paper_trading.py --update     # 先更新数据再运行
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pickle

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord


# ==================== 实盘模拟配置 ====================
PAPER_TRADING_CONFIG = {
    # 账户配置
    "account": 100000,  # 起始资金 10万元
    
    # 交易所配置 (全部使用 Qlib 原生参数)
    "exchange_kwargs": {
        "freq": "day",
        "limit_threshold": 0.095,      # 涨跌停限制 9.5%
        "deal_price": "open",          # 使用开盘价成交 (信号后第二天开盘)
        "open_cost": 0.0003,           # 买入手续费 万分之三
        "close_cost": 0.0003,          # 卖出手续费 万分之三
        "min_cost": 5,                 # 最低手续费 5元
        "impact_cost": 0.001,          # 滑点/冲击成本 0.1%
        "trade_unit": 100,             # 交易单位 100股/手
    },
    
    # 策略配置
    "strategy": {
        "topk": 10,      # 持有股票数量
        "n_drop": 2,     # 每天换仓数量
    },
    
    # 基准
    "benchmark": "SH000300",  # 沪深300指数
}


def update_data():
    """更新数据到最新"""
    print("=" * 60)
    print("📥 更新数据...")
    print("=" * 60)
    
    data_dir = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    
    # 方案1: 使用 crowd source 数据
    print("\n尝试从 chenditc/investment_data 下载最新数据...")
    
    try:
        # 下载最新数据
        subprocess.run([
            "wget", "-q", "--show-progress",
            "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz",
            "-O", "/tmp/qlib_bin.tar.gz"
        ], check=True)
        
        # 解压
        subprocess.run([
            "tar", "-zxf", "/tmp/qlib_bin.tar.gz",
            "-C", str(data_dir),
            "--strip-components=2"
        ], check=True)
        
        print("✅ 数据更新成功!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 数据更新失败: {e}")
        print("\n请手动执行以下命令:")
        print("  wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz")
        print(f"  tar -zxvf qlib_bin.tar.gz -C {data_dir} --strip-components=2")
        return False


def check_data_availability():
    """检查数据可用性"""
    calendar = D.calendar(start_time='2024-01-01', end_time='2025-12-31', freq='day')
    
    if calendar.size == 0:
        print("❌ 没有可用的交易日历数据")
        return None, None
    
    # 获取实际有数据的最后日期
    instruments = ['SH600000']
    fields = ['$close']
    data = D.features(instruments, fields, start_time='2025-01-01', end_time='2025-12-31', freq='day')
    data = data.dropna()
    
    if len(data) == 0:
        print("❌ 2025年没有股票数据")
        return None, None
    
    last_date = data.index[-1][1]
    print(f"📊 数据可用范围: 截止到 {last_date.strftime('%Y-%m-%d')}")
    
    return calendar, last_date


def run_paper_trading(start_date: str = None, end_date: str = None, model_type: str = "lightgbm"):
    """
    运行模拟实盘交易
    
    Parameters:
    -----------
    start_date : str
        回测开始日期，默认为数据可用的最近3个月
    end_date : str
        回测结束日期，默认为数据可用的最后日期
    model_type : str
        模型类型: "lightgbm" 或 "xgboost"
    """
    print("=" * 60)
    print("🚀 模拟实盘交易")
    print("=" * 60)
    
    # 初始化 Qlib
    provider_uri = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    print(f"✅ Qlib 初始化成功")
    
    # 检查数据
    calendar, last_date = check_data_availability()
    if last_date is None:
        return None
    
    # 设置日期范围
    if end_date is None:
        end_date = last_date.strftime('%Y-%m-%d')
    
    if start_date is None:
        # 默认使用最近6个月作为测试期
        start_dt = last_date - timedelta(days=180)
        start_date = start_dt.strftime('%Y-%m-%d')
    
    # 训练期设置
    train_end = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    train_start = (datetime.strptime(train_end, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
    valid_start = (datetime.strptime(train_end, '%Y-%m-%d') - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"\n📅 时间配置:")
    print(f"   训练期: {train_start} ~ {valid_start}")
    print(f"   验证期: {valid_start} ~ {train_end}")
    print(f"   回测期: {start_date} ~ {end_date}")
    
    print(f"\n💰 实盘配置:")
    print(f"   起始资金: ¥{PAPER_TRADING_CONFIG['account']:,}")
    print(f"   交易单位: {PAPER_TRADING_CONFIG['exchange_kwargs']['trade_unit']} 股/手")
    print(f"   买入手续费: {PAPER_TRADING_CONFIG['exchange_kwargs']['open_cost']*10000:.1f}‱")
    print(f"   卖出手续费: {PAPER_TRADING_CONFIG['exchange_kwargs']['close_cost']*10000:.1f}‱")
    print(f"   滑点成本: {PAPER_TRADING_CONFIG['exchange_kwargs']['impact_cost']*100:.2f}%")
    print(f"   成交价格: 开盘价 (信号后第二天)")
    print(f"   持仓数量: {PAPER_TRADING_CONFIG['strategy']['topk']} 只")
    
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
        exp_name = "PaperTrading_XGBoost"
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
        exp_name = "PaperTrading_LightGBM"
    
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
    
    # 运行实验
    print(f"\n🔬 训练 {model_type.upper()} 模型...")
    
    # 创建模型和数据集
    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)
    
    # 回测配置 - 使用实盘参数 (需要在模型创建后配置)
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
                "model": model,  # 传递模型
                "dataset": dataset,  # 传递数据集
                "topk": PAPER_TRADING_CONFIG["strategy"]["topk"],
                "n_drop": PAPER_TRADING_CONFIG["strategy"]["n_drop"],
            },
        },
        "backtest": {
            "start_time": start_date,
            "end_time": end_date,
            "account": PAPER_TRADING_CONFIG["account"],
            "benchmark": PAPER_TRADING_CONFIG["benchmark"],
            "exchange_kwargs": PAPER_TRADING_CONFIG["exchange_kwargs"],
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
        pred_df = recorder.load_object("pred.pkl")
        report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        analysis = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    
    # 输出结果
    print("\n" + "=" * 60)
    print("📊 模拟实盘结果")
    print("=" * 60)
    
    # 计算关键指标
    total_return = (report_df['account'].iloc[-1] / PAPER_TRADING_CONFIG["account"] - 1) * 100
    total_cost = report_df['cost'].sum() * 100
    total_turnover = report_df['turnover'].mean() * 100
    
    excess_with_cost = analysis.get("excess_return_with_cost", {})
    
    print(f"\n💵 账户表现:")
    print(f"   起始资金: ¥{PAPER_TRADING_CONFIG['account']:,.0f}")
    print(f"   最终资金: ¥{report_df['account'].iloc[-1]:,.0f}")
    print(f"   总收益率: {total_return:+.2f}%")
    print(f"   累计交易成本: {total_cost:.4f}%")
    print(f"   平均换手率: {total_turnover:.2f}%")
    
    if excess_with_cost:
        print(f"\n📈 风险指标 (扣费后):")
        print(f"   年化超额收益: {excess_with_cost.get('annualized_return', 0)*100:.2f}%")
        print(f"   信息比率: {excess_with_cost.get('information_ratio', 0):.4f}")
        print(f"   最大回撤: {excess_with_cost.get('max_drawdown', 0)*100:.2f}%")
    
    # 显示最新持仓
    dates = sorted(positions.keys())
    if dates:
        last_date = dates[-1]
        last_pos = positions[last_date]
        
        print(f"\n📋 最新持仓 ({last_date.strftime('%Y-%m-%d')}):")
        if hasattr(last_pos, 'position'):
            pos_dict = last_pos.position
            stocks = [(k, v) for k, v in pos_dict.items() 
                     if k not in ['cash', 'now_account_value'] and isinstance(v, dict)]
            
            print(f"   现金: ¥{pos_dict.get('cash', 0):,.2f}")
            print(f"   持仓股票数: {len(stocks)}")
            print(f"\n   {'股票代码':<12} {'持仓数量':>10} {'市值':>12} {'权重':>8}")
            print("   " + "-" * 50)
            
            for stock, info in sorted(stocks, key=lambda x: -x[1].get('weight', 0))[:10]:
                amount = info.get('amount', 0)
                price = info.get('price', 0)
                weight = info.get('weight', 0) * 100
                market_val = amount * price
                print(f"   {stock:<12} {amount:>10,.0f} ¥{market_val:>10,.0f} {weight:>7.2f}%")
    
    # 显示最近交易
    print(f"\n📅 最近5天交易记录:")
    print(f"   {'日期':<12} {'收益率':>10} {'换手率':>10} {'交易成本':>10}")
    print("   " + "-" * 50)
    for date, row in report_df.tail(5).iterrows():
        ret = row.get('return', 0) * 100
        turnover = row.get('turnover', 0) * 100
        cost = row.get('cost', 0) * 100
        print(f"   {date.strftime('%Y-%m-%d'):<12} {ret:>+9.4f}% {turnover:>9.2f}% {cost:>9.4f}%")
    
    # 保存结果
    output_dir = Path(__file__).parent / "output" / "paper_trading"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_df.to_csv(output_dir / "daily_report.csv")
    with open(output_dir / "positions.pkl", "wb") as f:
        pickle.dump(positions, f)
    
    print(f"\n✅ 结果已保存到: {output_dir}")
    
    return {
        "report": report_df,
        "positions": positions,
        "analysis": analysis,
    }


def main():
    parser = argparse.ArgumentParser(description="模拟实盘交易")
    parser.add_argument("--update", action="store_true", help="更新数据")
    parser.add_argument("--start", type=str, default=None, help="回测开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="回测结束日期 (YYYY-MM-DD)")
    parser.add_argument("--model", type=str, default="lightgbm", choices=["lightgbm", "xgboost"],
                       help="模型类型")
    
    args = parser.parse_args()
    
    if args.update:
        update_data()
    
    run_paper_trading(
        start_date=args.start,
        end_date=args.end,
        model_type=args.model
    )


if __name__ == "__main__":
    main()

