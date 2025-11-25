#!/usr/bin/env python3
"""
基于 qlib/examples/run_all_model.py 的简化测试脚本
用于验证数据准备和模型训练流程
"""
import sys
import os
from pathlib import Path

# 确保可以导入 qlib
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import qlib
from qlib.workflow import R
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.utils import init_instance_by_config
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord

from qlib_cn_lowfreq.config import QLIB_PROVIDER_URI, EXP_ROOT, ensure_directories

def test_workflow_with_existing_data():
    """使用本地已有数据测试完整工作流"""
    print("=" * 60)
    print("Qlib 工作流测试（基于 run_all_model.py 模式）")
    print("=" * 60)
    
    # 1. 初始化
    ensure_directories()
    print(f"\n[1] 初始化 Qlib，数据路径: {QLIB_PROVIDER_URI}")
    qlib.init(
        provider_uri=str(QLIB_PROVIDER_URI),
        region="cn",
        expression_cache=None,
        dataset_cache=None
    )
    
    # 2. 检查可用数据
    print("\n[2] 检查可用数据...")
    from qlib.data import D
    try:
        instruments = D.instruments("all")
        inst_list = D.list_instruments(instruments, as_list=True)
        print(f"  ✓ 可用股票数量: {len(inst_list)}")
        if inst_list:
            print(f"  样例: {inst_list[:5]}")
        
        # 检查数据日期范围
        calendar = D.calendar(start_time="2024-01-01", end_time="2024-12-31")
        print(f"  ✓ 交易日历: {len(calendar)} 个交易日")
        if len(calendar) > 0:
            print(f"  日期范围: {calendar[0]} 至 {calendar[-1]}")
    except Exception as e:
        print(f"  ✗ 数据检查失败: {e}")
        return False
    
    # 3. 构建数据集（使用 Alpha158 因子）
    print("\n[3] 构建数据集（Alpha158 因子）...")
    try:
        data_handler_config = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "instruments": "all",
                "start_time": "2024-01-01",
                "end_time": "2024-12-31",
                "freq": "day",
            },
        }
        handler = init_instance_by_config(data_handler_config)
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": ("2024-01-01", "2024-10-31"),
                "test": ("2024-11-01", "2024-12-31")
            }
        )
        print("  ✓ 数据集构建成功")
    except Exception as e:
        print(f"  ✗ 数据集构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 训练模型
    print("\n[4] 训练 LightGBM 模型...")
    try:
        model = LGBModel(
            loss="mse",
            num_leaves=64,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=100,  # 减少迭代次数以加快测试
        )
        
        with R.start(
            experiment_name="test_workflow",
            recorder_name="lightgbm_test",
            uri=str(EXP_ROOT.resolve())
        ) as raw_recorder:
            # 兼容不同版本的 recorder API
            if hasattr(raw_recorder, "log_params"):
                recorder = raw_recorder
            elif hasattr(raw_recorder, "active_recorder"):
                recorder = raw_recorder.active_recorder
            else:
                recorder = raw_recorder
            
            # 尝试记录参数
            try:
                if hasattr(recorder, "log_params"):
                    recorder.log_params({
                        "model": "LGBModel",
                        "num_leaves": 64,
                        "learning_rate": 0.05,
                        "n_estimators": 100
                    })
            except (AttributeError, TypeError):
                print("  警告: 无法记录参数（recorder API 不兼容）")
            
            print("  开始训练...")
            model.fit(dataset)
            print("  ✓ 模型训练完成")
            
            # 5. 预测
            print("\n[5] 生成预测...")
            pred = model.predict(dataset)
            print(f"  ✓ 预测完成，共 {len(pred)} 条记录")
            print(f"  预测样例:\n{pred.head()}")
            try:
                if hasattr(recorder, "log_metrics"):
                    recorder.log_metrics({"n_predictions": len(pred)})
            except (AttributeError, TypeError):
                pass
            
            # 6. 信号分析
            print("\n[6] 信号分析...")
            signal_record = SignalRecord(model, dataset, recorder)
            signal_record.generate()
            print("  ✓ 信号记录生成完成")
            
            sig_ana = SigAnaRecord(recorder)
            sig_ana.generate()
            print("  ✓ 信号分析完成")
            
            # 7. 回测（简化版，仅在有足够数据时运行）
            if len(inst_list) > 1:
                print("\n[7] 组合回测...")
                port_analysis_config = {
                    "strategy": {
                        "class": "TopkDropoutStrategy",
                        "module_path": "qlib.contrib.strategy.signal_strategy",
                        "kwargs": {"signal": "<PRED>", "topk": min(5, len(inst_list)), "n_drop": 1},
                    },
                    "executor": {
                        "class": "SimulatorExecutor",
                        "module_path": "qlib.backtest.executor",
                        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
                    },
                    "backtest": {
                        "start_time": "2024-11-01",
                        "end_time": "2024-12-31",
                        "account": 1e6,
                        "benchmark": inst_list[0] if inst_list else "SH000300",
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
                port_analyzer = PortAnaRecord(recorder, port_analysis_config, risk_analysis_freq="day")
                port_analyzer.generate()
                print("  ✓ 组合回测完成")
            else:
                print("\n[7] 跳过组合回测（数据不足，需要至少 2 只股票）")
            
            print(f"\n[8] 实验结果保存在: {EXP_ROOT}")
            print("  ✓ 测试完成！")
            
    except Exception as e:
        print(f"  ✗ 工作流执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_workflow_with_existing_data()
    sys.exit(0 if success else 1)

