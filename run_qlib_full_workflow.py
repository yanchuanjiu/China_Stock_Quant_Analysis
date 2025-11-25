#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qlib 完整工作流测试脚本
=====================
功能：
1. 下载 Qlib 官方中国股票数据（沪深300，日级数据）
2. 训练 LightGBM 和 XGBoost 模型
3. 执行回测并与 Buy&Hold 策略对比
4. 生成完整的分析报告和图表

运行方式：
    python run_qlib_full_workflow.py

结果查看位置：
    - experiments/ 目录：MLflow 实验记录
    - output/ 目录：回测报告、图表、模型效果对比
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# 使用系统安装的 qlib，不使用本地 qlib 源码
# 如果需要使用本地源码，取消下面注释：
# QLIB_PATH = Path(__file__).parent / "qlib"
# sys.path.insert(0, str(QLIB_PATH))

# ============================================================================
# 配置区
# ============================================================================
# 数据配置
PROVIDER_URI = Path.home() / ".qlib" / "qlib_data" / "cn_data"
EXPERIMENT_DIR = Path(__file__).parent / "output"
EXPERIMENT_DIR.mkdir(exist_ok=True)

# 时间配置 - 使用2024年数据进行测试
# 注意：Qlib 官方数据可能有延迟，这里使用可用的历史数据
DATA_START = "2020-01-01"
DATA_END = "2024-01-01"
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END = "2023-06-30"
TEST_START = "2023-07-01"
TEST_END = "2024-01-01"

# 市场和基准
MARKET = "csi300"
BENCHMARK = "SH000300"

# ============================================================================
# 数据下载
# ============================================================================
def download_data():
    """下载 Qlib 官方中国股票数据"""
    print("=" * 60)
    print("步骤 1: 下载 Qlib 官方中国股票数据")
    print("=" * 60)
    
    from qlib.tests.data import GetData
    
    target_dir = str(PROVIDER_URI)
    print(f"目标目录: {target_dir}")
    
    # 检查数据是否已存在
    if PROVIDER_URI.exists() and (PROVIDER_URI / "calendars").exists():
        print("✓ 数据已存在，跳过下载")
        return True
    
    print("正在从 Qlib 服务器下载数据...")
    print("（这可能需要几分钟，取决于网络速度）")
    
    try:
        GetData().qlib_data(
            target_dir=target_dir,
            region="cn",
            exists_skip=True
        )
        print("✓ 数据下载完成")
        return True
    except Exception as e:
        print(f"✗ 数据下载失败: {e}")
        print("\n尝试使用 crowd source 数据...")
        return download_crowd_source_data()


def download_crowd_source_data():
    """下载社区维护的数据（备选方案）"""
    import subprocess
    
    print("正在下载 crowd source 数据...")
    target_dir = str(PROVIDER_URI)
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # 下载数据
        subprocess.run([
            "wget", "-q", "--show-progress",
            "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz",
            "-O", "/tmp/qlib_bin.tar.gz"
        ], check=True)
        
        # 解压数据
        subprocess.run([
            "tar", "-zxvf", "/tmp/qlib_bin.tar.gz",
            "-C", target_dir, "--strip-components=2"
        ], check=True)
        
        print("✓ Crowd source 数据下载完成")
        return True
    except Exception as e:
        print(f"✗ Crowd source 数据下载失败: {e}")
        return False


# ============================================================================
# Qlib 初始化
# ============================================================================
def init_qlib():
    """初始化 Qlib"""
    print("\n" + "=" * 60)
    print("步骤 2: 初始化 Qlib")
    print("=" * 60)
    
    import qlib
    from qlib.constant import REG_CN
    
    qlib.init(
        provider_uri=str(PROVIDER_URI),
        region=REG_CN,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": "file:" + str(EXPERIMENT_DIR / "mlruns"),
                "default_exp_name": "qlib_workflow",
            },
        },
    )
    print(f"✓ Qlib 初始化完成")
    print(f"  数据路径: {PROVIDER_URI}")
    print(f"  实验目录: {EXPERIMENT_DIR}")


# ============================================================================
# 数据验证
# ============================================================================
def validate_data():
    """验证数据可用性"""
    print("\n" + "=" * 60)
    print("步骤 3: 验证数据可用性")
    print("=" * 60)
    
    from qlib.data import D
    
    # 获取交易日历
    cal = D.calendar(start_time=DATA_START, end_time=DATA_END, freq="day")
    print(f"✓ 交易日历: {len(cal)} 个交易日")
    print(f"  起始日期: {cal[0]}")
    print(f"  结束日期: {cal[-1]}")
    
    # 获取 CSI300 成分股
    instruments = D.instruments(market=MARKET)
    stock_list = D.list_instruments(
        instruments=instruments,
        start_time=DATA_START,
        end_time=DATA_END,
        as_list=True
    )
    print(f"✓ {MARKET.upper()} 成分股: {len(stock_list)} 只")
    print(f"  示例: {stock_list[:5]}")
    
    # 获取特征数据示例
    sample_features = D.features(
        stock_list[:3],
        ["$close", "$volume", "$high", "$low"],
        start_time=DATA_START,
        end_time=DATA_END,
        freq="day"
    )
    print(f"✓ 特征数据验证成功")
    print(f"  数据形状: {sample_features.shape}")
    
    return len(cal) > 0 and len(stock_list) > 0


# ============================================================================
# 模型训练
# ============================================================================
def get_model_configs():
    """获取模型配置"""
    
    # LightGBM 配置
    lgb_config = {
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
            "num_threads": 4,
            "early_stopping_rounds": 50,
            "num_boost_round": 500,
        },
    }
    
    # XGBoost 配置
    xgb_config = {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
        "kwargs": {
            "eval_metric": "rmse",
            "colsample_bytree": 0.8879,
            "eta": 0.05,
            "max_depth": 8,
            "n_estimators": 500,
            "subsample": 0.8789,
            "nthread": 4,
            "early_stopping_rounds": 50,
        },
    }
    
    return {
        "LightGBM": lgb_config,
        "XGBoost": xgb_config,
    }


def get_dataset_config():
    """获取数据集配置"""
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": DATA_START,
                    "end_time": DATA_END,
                    "fit_start_time": TRAIN_START,
                    "fit_end_time": TRAIN_END,
                    "instruments": MARKET,
                },
            },
            "segments": {
                "train": (TRAIN_START, TRAIN_END),
                "valid": (VALID_START, VALID_END),
                "test": (TEST_START, TEST_END),
            },
        },
    }


def get_backtest_config():
    """获取回测配置"""
    return {
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
                "signal": "<PRED>",
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": TEST_START,
            "end_time": TEST_END,
            "account": 100000000,
            "benchmark": BENCHMARK,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }


def train_and_evaluate(model_name, model_config, dataset_config, backtest_config):
    """训练单个模型并评估"""
    print(f"\n{'─' * 40}")
    print(f"训练模型: {model_name}")
    print(f"{'─' * 40}")
    
    from qlib.utils import init_instance_by_config, flatten_dict
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
    
    # 初始化模型和数据集
    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)
    
    # 更新回测配置中的信号
    bt_config = backtest_config.copy()
    bt_config["strategy"]["kwargs"]["signal"] = (model, dataset)
    
    results = {}
    
    with R.start(experiment_name=f"workflow_{model_name}"):
        # 记录参数
        R.log_params(model_name=model_name)
        R.log_params(train_period=f"{TRAIN_START} to {TRAIN_END}")
        R.log_params(test_period=f"{TEST_START} to {TEST_END}")
        R.log_params(market=MARKET)
        
        # 训练模型
        print(f"  正在训练 {model_name}...")
        model.fit(dataset)
        R.save_objects(**{f"{model_name}_model.pkl": model})
        print(f"  ✓ 模型训练完成")
        
        # 获取 recorder
        recorder = R.get_recorder()
        
        # 生成预测信号
        print(f"  正在生成预测信号...")
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        print(f"  ✓ 预测信号生成完成")
        
        # 信号分析
        print(f"  正在进行信号分析...")
        sar = SigAnaRecord(recorder)
        sar.generate()
        print(f"  ✓ 信号分析完成")
        
        # 回测
        print(f"  正在执行回测...")
        par = PortAnaRecord(recorder, bt_config, "day")
        par.generate()
        print(f"  ✓ 回测完成")
        
        # 获取指标
        metrics = recorder.list_metrics()
        results["metrics"] = metrics
        results["recorder_id"] = recorder.id
        
    return results


def run_all_models():
    """运行所有模型"""
    print("\n" + "=" * 60)
    print("步骤 4: 训练模型并执行回测")
    print("=" * 60)
    
    model_configs = get_model_configs()
    dataset_config = get_dataset_config()
    backtest_config = get_backtest_config()
    
    all_results = {}
    
    for model_name, model_config in model_configs.items():
        try:
            results = train_and_evaluate(
                model_name, model_config, dataset_config, backtest_config
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"  ✗ {model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
    
    return all_results


# ============================================================================
# 结果分析和报告
# ============================================================================
def generate_report(all_results):
    """生成分析报告"""
    print("\n" + "=" * 60)
    print("步骤 5: 生成分析报告")
    print("=" * 60)
    
    report_path = EXPERIMENT_DIR / "model_comparison_report.md"
    
    report = []
    report.append("# Qlib 模型效果对比报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n## 实验配置")
    report.append(f"- 市场: {MARKET.upper()}")
    report.append(f"- 基准: {BENCHMARK}")
    report.append(f"- 训练期: {TRAIN_START} ~ {TRAIN_END}")
    report.append(f"- 验证期: {VALID_START} ~ {VALID_END}")
    report.append(f"- 测试期: {TEST_START} ~ {TEST_END}")
    
    report.append("\n## 模型效果对比")
    report.append("\n### 信号质量指标")
    report.append("\n| 模型 | IC | ICIR | Rank IC | Rank ICIR |")
    report.append("|------|-----|------|---------|-----------|")
    
    for model_name, results in all_results.items():
        if "metrics" in results:
            m = results["metrics"]
            ic = m.get("IC", "N/A")
            icir = m.get("ICIR", "N/A")
            ric = m.get("Rank IC", "N/A")
            ricir = m.get("Rank ICIR", "N/A")
            
            ic_str = f"{ic:.4f}" if isinstance(ic, (int, float)) else ic
            icir_str = f"{icir:.4f}" if isinstance(icir, (int, float)) else icir
            ric_str = f"{ric:.4f}" if isinstance(ric, (int, float)) else ric
            ricir_str = f"{ricir:.4f}" if isinstance(ricir, (int, float)) else ricir
            
            report.append(f"| {model_name} | {ic_str} | {icir_str} | {ric_str} | {ricir_str} |")
    
    report.append("\n### 回测绩效指标")
    report.append("\n| 模型 | 年化收益 | 信息比率 | 最大回撤 | 夏普比率 |")
    report.append("|------|----------|----------|----------|----------|")
    
    for model_name, results in all_results.items():
        if "metrics" in results:
            m = results["metrics"]
            ar = m.get("1day.excess_return_with_cost.annualized_return", 
                      m.get("annualized_return", "N/A"))
            ir = m.get("1day.excess_return_with_cost.information_ratio",
                      m.get("information_ratio", "N/A"))
            md = m.get("1day.excess_return_with_cost.max_drawdown",
                      m.get("max_drawdown", "N/A"))
            sr = m.get("1day.excess_return_with_cost.sharpe_ratio",
                      m.get("sharpe_ratio", "N/A"))
            
            ar_str = f"{ar*100:.2f}%" if isinstance(ar, (int, float)) else ar
            ir_str = f"{ir:.4f}" if isinstance(ir, (int, float)) else ir
            md_str = f"{md*100:.2f}%" if isinstance(md, (int, float)) else md
            sr_str = f"{sr:.4f}" if isinstance(sr, (int, float)) else sr
            
            report.append(f"| {model_name} | {ar_str} | {ir_str} | {md_str} | {sr_str} |")
    
    report.append("\n## 指标说明")
    report.append("""
- **IC (Information Coefficient)**: 预测值与实际收益的相关系数，越高越好
- **ICIR (IC Information Ratio)**: IC 的均值除以标准差，衡量 IC 的稳定性
- **Rank IC**: 排名相关系数，对异常值更鲁棒
- **年化收益**: 策略相对于基准的年化超额收益
- **信息比率**: 超额收益与跟踪误差的比值
- **最大回撤**: 策略的最大亏损幅度
- **夏普比率**: 风险调整后收益
""")
    
    report.append("\n## 结果查看位置")
    report.append(f"""
1. **MLflow 实验记录**: `{EXPERIMENT_DIR}/mlruns/`
   - 使用 `mlflow ui --backend-store-uri {EXPERIMENT_DIR}/mlruns` 启动 Web UI
   
2. **模型文件**: 保存在 MLflow artifacts 中

3. **预测结果**: `pred.pkl` 和 `label.pkl` 在 artifacts 目录

4. **信号分析**: `sig_analysis/` 目录包含 IC 时序数据

5. **回测报告**: `portfolio_analysis/` 目录包含详细回测结果
""")
    
    # 写入报告
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"✓ 报告已保存到: {report_path}")
    
    # 打印到控制台
    print("\n" + "=" * 60)
    print("模型效果对比总结")
    print("=" * 60)
    for line in report[report.index("### 信号质量指标"):]:
        print(line)
    
    return report_path


def print_result_locations():
    """打印结果查看位置"""
    print("\n" + "=" * 60)
    print("结果查看位置")
    print("=" * 60)
    print(f"""
📁 实验输出目录: {EXPERIMENT_DIR}

📊 查看实验结果的方式:

1. 【MLflow Web UI】推荐
   运行命令: mlflow ui --backend-store-uri file:{EXPERIMENT_DIR}/mlruns
   然后访问: http://localhost:5000
   
2. 【模型对比报告】
   文件位置: {EXPERIMENT_DIR}/model_comparison_report.md
   
3. 【预测结果】
   位置: {EXPERIMENT_DIR}/mlruns/<experiment_id>/<run_id>/artifacts/
   - pred.pkl: 模型预测值
   - label.pkl: 实际标签
   
4. 【信号分析】
   位置: artifacts/sig_analysis/
   - ic.pkl: IC 时序数据
   - ric.pkl: Rank IC 时序数据
   
5. 【回测详情】
   位置: artifacts/portfolio_analysis/
   - 包含每日持仓、交易记录、收益曲线等

💡 提示: 
   - IC > 0.03 通常被认为是有效信号
   - ICIR > 0.5 表示信号稳定
   - 信息比率 > 1 表示策略优于基准
""")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """主函数"""
    print("=" * 60)
    print("Qlib 完整工作流测试")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 下载数据
    if not download_data():
        print("\n✗ 数据下载失败，无法继续")
        return
    
    # 2. 初始化 Qlib
    init_qlib()
    
    # 3. 验证数据
    if not validate_data():
        print("\n✗ 数据验证失败，无法继续")
        return
    
    # 4. 训练模型并回测
    all_results = run_all_models()
    
    if not all_results:
        print("\n✗ 没有模型训练成功")
        return
    
    # 5. 生成报告
    generate_report(all_results)
    
    # 6. 打印结果位置
    print_result_locations()
    
    print("\n" + "=" * 60)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
