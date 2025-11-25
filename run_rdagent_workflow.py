#!/usr/bin/env python3
"""
RD-Agent 完整闭环测试脚本
============================
功能：
1. 数据获取：CSI300 沪深300股票，2024年至今的日级交易数据
2. 模型训练：XGBoost 和 LightGBM
3. 回测分析：TopkDropout 策略 vs Buy&Hold 基准对比
4. 结果展示：IC/ICIR/收益率/最大回撤等指标对比图表

结果输出位置：
- experiments/ 目录：MLflow 实验记录
- output/ 目录：图表和指标汇总
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# 使用已安装的 pyqlib 而非本地 qlib 源码

import pandas as pd
import numpy as np

# Qlib imports
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord


def setup_output_dir():
    """创建输出目录"""
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def init_qlib():
    """初始化 Qlib"""
    provider_uri = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    
    if not provider_uri.exists():
        print(f"❌ 数据目录不存在: {provider_uri}")
        print("请先运行: python qlib/scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn")
        sys.exit(1)
    
    exp_dir = PROJECT_ROOT / "experiments"
    exp_dir.mkdir(exist_ok=True)
    
    qlib.init(
        provider_uri=str(provider_uri),
        region=REG_CN,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": "file:" + str(exp_dir),
                "default_exp_name": "RDAgent_Workflow",
            },
        }
    )
    print(f"✅ Qlib 初始化成功，数据路径: {provider_uri}")
    return provider_uri


def check_data_availability():
    """检查数据可用性"""
    print("\n📊 检查数据可用性...")
    
    # 获取日历
    calendar = D.calendar(start_time='2024-01-01', end_time='2025-12-31', freq='day')
    print(f"  交易日历: {len(calendar)} 个交易日")
    print(f"  起始日期: {calendar[0] if len(calendar) > 0 else 'N/A'}")
    print(f"  结束日期: {calendar[-1] if len(calendar) > 0 else 'N/A'}")
    
    # 获取 CSI300 成分股
    instruments = D.instruments(market='csi300')
    stock_list = D.list_instruments(
        instruments=instruments, 
        start_time='2024-01-01', 
        end_time='2025-12-31', 
        as_list=True
    )
    print(f"  CSI300 成分股: {len(stock_list)} 只")
    
    # 检查数据完整性
    sample_stocks = stock_list[:5]
    fields = ['$close', '$open', '$high', '$low', '$volume']
    sample_data = D.features(
        sample_stocks, 
        fields, 
        start_time='2024-01-01', 
        end_time='2025-12-31', 
        freq='day'
    )
    print(f"  样本数据形状: {sample_data.shape}")
    
    return calendar, stock_list


def get_model_config(model_type: str, start_time: str, end_time: str):
    """获取模型配置"""
    
    # 数据处理配置
    data_handler_config = {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": start_time,
        "fit_end_time": "2024-09-30",  # 训练集截止日期
        "instruments": "csi300",
    }
    
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
                "topk": 30,
                "n_drop": 3,
            },
        },
        "backtest": {
            "start_time": "2024-10-08",  # 回测开始日期（国庆后）
            "end_time": end_time,
            "account": 100000000,
            "benchmark": "SH000300",  # 沪深300指数
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,  # 涨跌停限制
                "deal_price": "close",
                "open_cost": 0.0005,  # 买入手续费
                "close_cost": 0.0015,  # 卖出手续费（含印花税）
                "min_cost": 5,
            },
        },
    }
    
    # 模型配置
    if model_type == "xgboost":
        model_config = {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
            "kwargs": {
                "eval_metric": "rmse",
                "colsample_bytree": 0.8879,
                "eta": 0.0421,
                "max_depth": 8,
                "n_estimators": 500,
                "subsample": 0.8789,
                "nthread": 8,
            }
        }
    elif model_type == "lightgbm":
        model_config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.05,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 8,
            }
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
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
                "train": (start_time, "2024-06-30"),
                "valid": ("2024-07-01", "2024-09-30"),
                "test": ("2024-10-08", end_time),  # 国庆后开始测试
            },
        },
    }
    
    return {
        "model": model_config,
        "dataset": dataset_config,
        "port_analysis_config": port_analysis_config,
    }


def run_model_experiment(model_type: str, config: dict, output_dir: Path):
    """运行单个模型实验"""
    print(f"\n🚀 运行 {model_type.upper()} 模型实验...")
    
    # 初始化模型和数据集
    model = init_instance_by_config(config["model"])
    dataset = init_instance_by_config(config["dataset"])
    
    # 更新策略配置中的信号源
    port_analysis_config = config["port_analysis_config"].copy()
    port_analysis_config["strategy"]["kwargs"]["signal"] = (model, dataset)
    
    # 打印数据集信息
    train_data = dataset.prepare("train")
    valid_data = dataset.prepare("valid")
    test_data = dataset.prepare("test")
    
    print(f"  训练集: {train_data.shape}")
    print(f"  验证集: {valid_data.shape}")
    print(f"  测试集: {test_data.shape}")
    
    results = {}
    
    # 开始实验记录
    with R.start(experiment_name=f"RDAgent_{model_type}"):
        # 记录参数
        R.log_params(**flatten_dict(config["model"]))
        R.log_params(model_type=model_type)
        
        # 训练模型
        print(f"  训练模型...")
        model.fit(dataset)
        
        # 保存模型
        R.save_objects(**{f"{model_type}_model.pkl": model})
        
        # 获取记录器
        recorder = R.get_recorder()
        
        # 生成预测信号
        print(f"  生成预测信号...")
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        
        # 信号分析
        print(f"  信号分析...")
        sar = SigAnaRecord(recorder)
        sar.generate()
        
        # 回测分析
        print(f"  回测分析...")
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        
        # 获取指标
        metrics = recorder.list_metrics()
        results["metrics"] = metrics
        results["recorder_id"] = recorder.id
        
        print(f"\n📈 {model_type.upper()} 模型指标:")
        print(f"  IC: {metrics.get('IC', 'N/A'):.4f}")
        print(f"  ICIR: {metrics.get('ICIR', 'N/A'):.4f}")
        print(f"  Rank IC: {metrics.get('Rank IC', 'N/A'):.4f}")
        print(f"  Rank ICIR: {metrics.get('Rank ICIR', 'N/A'):.4f}")
        
        # 获取回测指标
        for key, value in metrics.items():
            if 'annualized_return' in key:
                print(f"  年化收益率: {value:.4f}")
            if 'information_ratio' in key:
                print(f"  信息比率: {value:.4f}")
            if 'max_drawdown' in key:
                print(f"  最大回撤: {value:.4f}")
    
    return results


def generate_comparison_report(results: dict, output_dir: Path):
    """生成模型对比报告"""
    print("\n📊 生成模型对比报告...")
    
    # 创建对比表格
    comparison_data = []
    
    for model_type, result in results.items():
        metrics = result.get("metrics", {})
        row = {
            "模型": model_type.upper(),
            "IC": metrics.get("IC", 0),
            "ICIR": metrics.get("ICIR", 0),
            "Rank IC": metrics.get("Rank IC", 0),
            "Rank ICIR": metrics.get("Rank ICIR", 0),
        }
        
        # 添加回测指标
        for key, value in metrics.items():
            if 'annualized_return' in key and 'excess' in key:
                row["超额年化收益"] = value
            elif 'information_ratio' in key and 'excess' in key:
                row["信息比率"] = value
            elif 'max_drawdown' in key and 'excess' in key:
                row["最大回撤"] = value
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # 保存到 CSV
    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  对比结果已保存: {csv_path}")
    
    # 生成 Markdown 报告
    report_path = output_dir / "EXPERIMENT_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# RD-Agent 量化实验报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验配置\n\n")
        f.write("- **数据范围**: 2024-01-01 至今\n")
        f.write("- **股票池**: 沪深300 (CSI300)\n")
        f.write("- **数据频率**: 日级\n")
        f.write("- **特征**: Alpha158 因子库\n")
        f.write("- **训练集**: 2024-01-01 ~ 2024-06-30\n")
        f.write("- **验证集**: 2024-07-01 ~ 2024-09-30\n")
        f.write("- **测试集**: 2024-10-01 ~ 至今\n")
        f.write("- **策略**: TopkDropout (Top30, Drop3)\n")
        f.write("- **基准**: 沪深300指数 (SH000300)\n\n")
        
        f.write("## 模型对比结果\n\n")
        f.write("| 模型 | IC | ICIR | Rank IC | Rank ICIR | 超额年化收益 | 信息比率 | 最大回撤 |\n")
        f.write("|------|-----|------|---------|-----------|--------------|----------|----------|\n")
        
        for _, row in df.iterrows():
            f.write(f"| {row['模型']} | ")
            f.write(f"{row.get('IC', 0):.4f} | ")
            f.write(f"{row.get('ICIR', 0):.4f} | ")
            f.write(f"{row.get('Rank IC', 0):.4f} | ")
            f.write(f"{row.get('Rank ICIR', 0):.4f} | ")
            f.write(f"{row.get('超额年化收益', 0):.4f} | ")
            f.write(f"{row.get('信息比率', 0):.4f} | ")
            f.write(f"{row.get('最大回撤', 0):.4f} |\n")
        
        f.write("\n## 指标说明\n\n")
        f.write("- **IC (Information Coefficient)**: 预测值与实际收益的相关系数，越高越好\n")
        f.write("- **ICIR**: IC的均值/标准差，衡量预测稳定性，越高越好\n")
        f.write("- **Rank IC**: 排序相关系数，对异常值更稳健\n")
        f.write("- **Rank ICIR**: Rank IC的稳定性指标\n")
        f.write("- **超额年化收益**: 相对于基准的年化超额收益\n")
        f.write("- **信息比率**: 超额收益/跟踪误差，越高越好\n")
        f.write("- **最大回撤**: 最大净值回撤幅度，越小越好\n\n")
        
        f.write("## 结果位置\n\n")
        f.write("- 实验记录: `experiments/` 目录 (MLflow 格式)\n")
        f.write("- 对比报告: `output/EXPERIMENT_REPORT.md`\n")
        f.write("- 数据表格: `output/model_comparison.csv`\n\n")
        
        f.write("## 如何查看详细结果\n\n")
        f.write("```bash\n")
        f.write("# 启动 MLflow UI 查看详细实验记录\n")
        f.write("cd experiments && mlflow ui\n")
        f.write("# 然后访问 http://localhost:5000\n")
        f.write("```\n")
    
    print(f"  实验报告已保存: {report_path}")
    
    return df


def main():
    """主函数"""
    print("=" * 60)
    print("🔬 RD-Agent 量化分析完整闭环测试")
    print("=" * 60)
    
    # 设置输出目录
    output_dir = setup_output_dir()
    
    # 初始化 Qlib
    init_qlib()
    
    # 检查数据可用性
    calendar, stock_list = check_data_availability()
    
    # 获取实际数据的结束日期
    # 注意：crowd source 数据只更新到 2025-05-14
    end_time = '2025-05-14'  # 使用数据实际可用的结束日期
    start_time = '2024-01-01'
    
    print(f"\n📅 实验时间范围: {start_time} ~ {end_time}")
    
    # 存储所有模型结果
    all_results = {}
    
    # 运行 XGBoost 模型
    try:
        xgb_config = get_model_config("xgboost", start_time, end_time)
        all_results["xgboost"] = run_model_experiment("xgboost", xgb_config, output_dir)
    except Exception as e:
        print(f"❌ XGBoost 实验失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 运行 LightGBM 模型
    try:
        lgb_config = get_model_config("lightgbm", start_time, end_time)
        all_results["lightgbm"] = run_model_experiment("lightgbm", lgb_config, output_dir)
    except Exception as e:
        print(f"❌ LightGBM 实验失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 生成对比报告
    if all_results:
        comparison_df = generate_comparison_report(all_results, output_dir)
        
        print("\n" + "=" * 60)
        print("📊 模型对比汇总")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✅ RD-Agent 完整闭环测试完成!")
    print("=" * 60)
    print(f"\n📁 结果输出位置:")
    print(f"   - 实验记录: {PROJECT_ROOT / 'experiments'}")
    print(f"   - 对比报告: {output_dir / 'EXPERIMENT_REPORT.md'}")
    print(f"   - 数据表格: {output_dir / 'model_comparison.csv'}")
    print(f"\n💡 查看详细实验记录:")
    print(f"   cd {PROJECT_ROOT / 'experiments'} && mlflow ui")


if __name__ == "__main__":
    main()

