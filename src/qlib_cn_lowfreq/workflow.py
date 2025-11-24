"""Low-frequency Qlib workflow: factor extraction, model training, and backtest."""
from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional, TYPE_CHECKING

from .config import DEFAULT_FREQ, EXP_ROOT, QLIB_PROVIDER_URI, ensure_directories

if TYPE_CHECKING:  # pragma: no cover
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP


def require_dependencies():
    try:
        qlib = importlib.import_module("qlib")
        contrib_data = importlib.import_module("qlib.contrib.data.handler")
        contrib_eval = importlib.import_module("qlib.contrib.eval")
        contrib_model = importlib.import_module("qlib.contrib.model.gbdt")
        contrib_strategy = importlib.import_module("qlib.contrib.strategy.signal_strategy")
        dataset_mod = importlib.import_module("qlib.data.dataset")
        dataset_handler_mod = importlib.import_module("qlib.data.dataset.handler")
        utils_mod = importlib.import_module("qlib.utils")
        workflow_mod = importlib.import_module("qlib.workflow")
        workflow_task_mod = importlib.import_module("qlib.workflow.task")
        workflow_records_mod = importlib.import_module("qlib.workflow.record_temp")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "缺少 pyqlib 依赖，无法运行训练与回测。请参考 README 通过离线 wheel 安装依赖。"
        ) from exc

    return {
        "qlib": qlib,
        "Alpha158": contrib_data.Alpha158,
        "risk_analysis": contrib_eval.risk_analysis,
        "LGBModel": contrib_model.LGBModel,
        "TopkDropoutStrategy": contrib_strategy.TopkDropoutStrategy,
        "DatasetH": dataset_mod.DatasetH,
        "DataHandlerLP": dataset_handler_mod.DataHandlerLP,
        "init_instance_by_config": utils_mod.init_instance_by_config,
        "R": workflow_mod.R,
        "Task": workflow_task_mod.Task,
        "SignalRecord": workflow_records_mod.SignalRecord,
        "SigAnaRecord": workflow_records_mod.SigAnaRecord,
        "PortAnaRecord": workflow_records_mod.PortAnaRecord,
    }


@dataclass
class WorkflowArgs:
    market: str
    start: str
    end: str
    freq: str = DEFAULT_FREQ
    horizon: int = 20
    topk: int = 30
    n_drop: int = 10
    benchmark: str = "SH000300"
    lookback_months: int = 3
    account: float = 1e8


def build_dataset(deps, market: str, start: str, end: str, freq: str):
    """Build a Qlib Dataset using Alpha158 handler for the given market."""

    DatasetH = deps["DatasetH"]
    DataHandlerLP = deps["DataHandlerLP"]
    init_instance_by_config = deps["init_instance_by_config"]

    data_handler_config = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "instruments": market,
            "start_time": start,
            "end_time": end,
            "freq": freq,
        },
    }
    handler: DataHandlerLP = init_instance_by_config(data_handler_config)
    dataset = DatasetH(handler=handler, segments={"train": (start, end), "test": (start, end)})
    return dataset


def run_experiment(args: WorkflowArgs) -> None:
    """Run the full Qlib experiment pipeline for low-frequency trading."""

    deps = require_dependencies()
    ensure_directories()
    qlib = deps["qlib"]
    qlib.init(provider_uri=str(QLIB_PROVIDER_URI), region="cn", expression_cache=None, dataset_cache=None)

    dataset = build_dataset(deps, args.market, start=args.start, end=args.end, freq=args.freq)

    LGBModel = deps["LGBModel"]
    TopkDropoutStrategy = deps["TopkDropoutStrategy"]
    Task = deps["Task"]
    risk_analysis = deps["risk_analysis"]
    R = deps["R"]
    SignalRecord = deps["SignalRecord"]
    SigAnaRecord = deps["SigAnaRecord"]
    PortAnaRecord = deps["PortAnaRecord"]

    model = LGBModel(
        loss="mse",
        num_leaves=64,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=200,
    )

    with R.start(exp_name="lowfreq_ths", recorder_name="model_train", uri=str(EXP_ROOT.resolve())) as recorder:
        recorder.log_params(vars(args))
        model.fit(dataset)
        pred = model.predict(dataset)
        recorder.log_metrics({"n_predictions": len(pred)})

        label = dataset.prepare("test", col_set="label")
        analysis_df = risk_analysis(pred=pred, label=label)
        recorder.log_df("risk_analysis", analysis_df)

        backtest_end = datetime.fromisoformat(args.end).date()
        lookback_start = backtest_end - timedelta(days=30 * args.lookback_months)
        backtest_start = max(datetime.fromisoformat(args.start).date(), lookback_start)

        signal_record = SignalRecord(model, dataset, recorder)
        signal_record.generate()

        SigAnaRecord(recorder).generate()

        port_analysis_config = {
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {"signal": "<PRED>", "topk": args.topk, "n_drop": args.n_drop},
            },
            "executor": {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
            },
            "backtest": {
                "start_time": backtest_start.strftime("%Y-%m-%d"),
                "end_time": backtest_end.strftime("%Y-%m-%d"),
                "account": args.account,
                "benchmark": args.benchmark,
                "exchange_kwargs": {
                    "freq": args.freq,
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
        }

        port_analyzer = PortAnaRecord(recorder, port_analysis_config, risk_analysis_freq=args.freq)
        port_analyzer.generate()

        if hasattr(port_analyzer, "all_freq") and port_analyzer.all_freq:
            freq_suffix = port_analyzer.all_freq[0]
            report_key = f"portfolio_analysis/report_normal_{freq_suffix}.pkl"
            try:
                report = recorder.load_object(report_key)
            except Exception:
                report = None

            if report is not None:
                strategy_return = (1 + (report["return"] - report.get("cost", 0.0))).prod() - 1
                benchmark_return = (1 + report["bench"]).prod() - 1
                recorder.log_metrics({
                    "strategy_total_return": strategy_return,
                    "benchmark_total_return": benchmark_return,
                    "outperformed_benchmark": strategy_return > benchmark_return,
                })


def parse_args(argv: Optional[Iterable[str]] = None) -> WorkflowArgs:
    parser = argparse.ArgumentParser(description="Low-frequency Qlib workflow on TongHuaShun data")
    parser.add_argument("--market", default="csi300", help="Qlib market name or instrument list")
    parser.add_argument("--start", required=True, help="Training start date")
    parser.add_argument("--end", required=True, help="Training end date")
    parser.add_argument("--freq", default=DEFAULT_FREQ, help="Data frequency, default is daily")
    parser.add_argument("--horizon", type=int, default=20, help="Forecast horizon in trading days")
    parser.add_argument("--topk", type=int, default=30, help="Top-K stocks selected each rebalance")
    parser.add_argument("--n-drop", type=int, default=10, dest="n_drop", help="Number of stocks dropped each round")
    parser.add_argument("--benchmark", default="SH000300", help="Benchmark symbol for Buy & Hold comparison")
    parser.add_argument("--lookback-months", type=int, default=3, dest="lookback_months", help="Backtest window length in months")
    parser.add_argument("--account", type=float, default=1e8, help="Initial account value for backtesting")
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    return WorkflowArgs(
        market=parsed.market,
        start=parsed.start,
        end=parsed.end,
        freq=parsed.freq,
        horizon=parsed.horizon,
        topk=parsed.topk,
        n_drop=parsed.n_drop,
        benchmark=parsed.benchmark,
        lookback_months=parsed.lookback_months,
        account=parsed.account,
    )


if __name__ == "__main__":  # pragma: no cover
    workflow_args = parse_args()
    run_experiment(workflow_args)
