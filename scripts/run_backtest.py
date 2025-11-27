#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一回测入口

用法:
    python scripts/run_backtest.py --config configs/strategies/volume_factor_lgb.yaml
    python scripts/run_backtest.py --compare all
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord


def load_config(config_path: str) -> dict:
    """加载策略配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_single_backtest(config: dict) -> dict:
    """
    运行单个策略回测
    """
    strategy_name = config['strategy']['name']
    data_config = config['data']
    model_config = config['model']
    backtest_config = config['backtest']
    
    print(f"\n{'='*70}")
    print(f"🚀 回测策略: {strategy_name}")
    print(f"{'='*70}")
    
    # 数据处理器配置
    handler_config = {
        "class": config['factors']['handler'],
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": data_config['train_start'],
            "end_time": data_config['test_end'],
            "fit_start_time": data_config['train_start'],
            "fit_end_time": data_config['train_end'],
            "instruments": data_config['instruments'],
        },
    }
    
    # 数据集配置
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": (data_config['train_start'], data_config['valid_start']),
                "valid": (data_config['valid_start'], data_config['valid_end']),
                "test": (data_config['test_start'], data_config['test_end']),
            },
        },
    }
    
    print(f"  训练期: {data_config['train_start']} ~ {data_config['valid_start']}")
    print(f"  验证期: {data_config['valid_start']} ~ {data_config['valid_end']}")
    print(f"  测试期: {data_config['test_start']} ~ {data_config['test_end']}")
    
    # 初始化模型和数据集
    model = init_instance_by_config({
        "class": model_config['class'],
        "module_path": model_config['module_path'],
        "kwargs": model_config['params'],
    })
    dataset = init_instance_by_config(dataset_config)
    
    # 回测配置
    strategy_config = backtest_config['strategy']
    exchange_config = backtest_config['exchange']
    
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
                "verbose": True,
            },
        },
        "strategy": {
            "class": strategy_config['class'],
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "model": model,
                "dataset": dataset,
                "topk": strategy_config['topk'],
                "n_drop": strategy_config['n_drop'],
            },
        },
        "backtest": {
            "start_time": data_config['test_start'],
            "end_time": data_config['test_end'],
            "account": backtest_config['account'],
            "benchmark": data_config['benchmark'],
            "exchange_kwargs": exchange_config,
        },
    }
    
    # 实验目录
    exp_dir = PROJECT_ROOT / "experiments"
    os.environ["MLFLOW_TRACKING_URI"] = str(exp_dir)
    
    with R.start(experiment_name=strategy_name):
        # 训练
        print("  训练模型...")
        model.fit(dataset)
        
        recorder = R.get_recorder()
        
        # 生成信号
        print("  生成预测信号...")
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        
        # 回测
        print("  执行回测...")
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        
        # 加载结果
        report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        analysis = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    
    # 计算指标
    returns = report_df['return'].dropna()
    account = backtest_config['account']
    
    total_ret = (report_df['account'].iloc[-1] / account - 1) * 100
    ann_ret = ((1 + total_ret/100) ** (252/len(returns)) - 1) * 100 if len(returns) > 0 else 0
    ann_vol = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() - 0.03/252) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    max_dd = ((cumulative - cumulative.cummax()) / cumulative.cummax()).min() * 100
    
    result = {
        'strategy': strategy_name,
        'total_return': total_ret,
        'annualized_return': ann_ret,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'final_value': report_df['account'].iloc[-1],
        'trading_days': len(returns),
        'report': report_df,
        'positions': positions,
        'analysis': analysis,
    }
    
    print(f"\n📊 回测结果:")
    print(f"  总收益率: {total_ret:+.2f}%")
    print(f"  年化收益: {ann_ret:+.2f}%")
    print(f"  夏普比率: {sharpe:.4f}")
    print(f"  最大回撤: {max_dd:.2f}%")
    
    return result


def compare_strategies(config_dir: str) -> pd.DataFrame:
    """
    对比多个策略
    """
    config_dir = Path(config_dir)
    config_files = list(config_dir.glob("*.yaml"))
    
    if not config_files:
        print(f"❌ 未找到配置文件: {config_dir}")
        return pd.DataFrame()
    
    results = []
    for config_file in config_files:
        try:
            config = load_config(str(config_file))
            result = run_single_backtest(config)
            results.append({
                'strategy': result['strategy'],
                'total_return': result['total_return'],
                'annualized_return': result['annualized_return'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown'],
                'final_value': result['final_value'],
            })
        except Exception as e:
            print(f"❌ {config_file.name} 失败: {e}")
    
    df = pd.DataFrame(results)
    df = df.sort_values('sharpe_ratio', ascending=False)
    
    print("\n" + "=" * 80)
    print("📊 策略对比")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # 保存结果
    output_path = PROJECT_ROOT / "output" / "reports" / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ 结果已保存: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="统一回测入口")
    parser.add_argument("--config", type=str, help="策略配置文件路径")
    parser.add_argument("--compare", action="store_true", help="对比所有策略")
    parser.add_argument("--config-dir", type=str, default="configs/strategies", help="策略配置目录")
    
    args = parser.parse_args()
    
    # 初始化 Qlib
    provider_uri = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    
    if args.compare:
        config_dir = PROJECT_ROOT / args.config_dir
        compare_strategies(str(config_dir))
    elif args.config:
        config = load_config(args.config)
        run_single_backtest(config)
    else:
        # 默认运行量比因子策略
        default_config = PROJECT_ROOT / "configs" / "strategies" / "volume_factor_lgb.yaml"
        if default_config.exists():
            config = load_config(str(default_config))
            run_single_backtest(config)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()

