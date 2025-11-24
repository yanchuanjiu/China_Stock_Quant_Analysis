"""Low-frequency Qlib workflow: factor extraction, model training, and backtest."""
from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
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

    model = LGBModel(
        loss="mse",
        num_leaves=64,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=200,
    )

    strategy = TopkDropoutStrategy(
        signal="score",
        topk=args.topk,
        n_drop=args.n_drop,
    )

    task = Task(model=model, dataset=dataset, strategy=strategy, recorder=None)

    with R.start(exp_name="lowfreq_ths", recorder_name="model_train", uri=str(EXP_ROOT.resolve())) as recorder:
        recorder.log_params(vars(args))
        model.fit(dataset)
        pred = model.predict(dataset)
        recorder.log_metrics({"n_predictions": len(pred)})

        label = dataset.prepare("test", col_set="label")
        analysis_df = risk_analysis(pred=pred, label=label)
        recorder.log_df("risk_analysis", analysis_df)


def parse_args(argv: Optional[Iterable[str]] = None) -> WorkflowArgs:
    parser = argparse.ArgumentParser(description="Low-frequency Qlib workflow on TongHuaShun data")
    parser.add_argument("--market", default="csi300", help="Qlib market name or instrument list")
    parser.add_argument("--start", required=True, help="Training start date")
    parser.add_argument("--end", required=True, help="Training end date")
    parser.add_argument("--freq", default=DEFAULT_FREQ, help="Data frequency, default is daily")
    parser.add_argument("--horizon", type=int, default=20, help="Forecast horizon in trading days")
    parser.add_argument("--topk", type=int, default=30, help="Top-K stocks selected each rebalance")
    parser.add_argument("--n-drop", type=int, default=10, dest="n_drop", help="Number of stocks dropped each round")
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    return WorkflowArgs(
        market=parsed.market,
        start=parsed.start,
        end=parsed.end,
        freq=parsed.freq,
        horizon=parsed.horizon,
        topk=parsed.topk,
        n_drop=parsed.n_drop,
    )


if __name__ == "__main__":
    workflow_args = parse_args()
    run_experiment(workflow_args)
