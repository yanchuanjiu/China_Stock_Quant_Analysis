import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qlib_cn_lowfreq import workflow


class DummyDataset:
    def __init__(self, handler=None, segments=None):
        self.handler = handler
        self.segments = segments

    def prepare(self, *_args, **_kwargs):
        return [0.0]


class DummyModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fitted = False

    def fit(self, dataset):
        self.fitted = True
        self.dataset = dataset

    def predict(self, _dataset):
        return [1.0, 2.0]


class DummyStrategy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyTask:
    def __init__(self, model=None, dataset=None, strategy=None, recorder=None):
        self.model = model
        self.dataset = dataset
        self.strategy = strategy
        self.recorder = recorder


class DummyRecorder:
    def __init__(self, sink):
        self.sink = sink

    def log_params(self, params):
        self.sink["params"] = params

    def log_metrics(self, metrics):
        self.sink["metrics"] = metrics

    def log_df(self, name, df):
        self.sink[name] = df


class DummyR:
    def __init__(self, sink):
        self.sink = sink

    def start(self, *args, **kwargs):
        recorder = DummyRecorder(self.sink)

        class Ctx:
            def __enter__(self_nonlocal):
                return recorder

            def __exit__(self_nonlocal, exc_type, exc, tb):
                return False

        return Ctx()


def build_stub_dependencies(sink):
    return {
        "qlib": mock.Mock(init=mock.Mock()),
        "Alpha158": mock.Mock(),
        "risk_analysis": mock.Mock(return_value={"risk": 1.0}),
        "LGBModel": DummyModel,
        "TopkDropoutStrategy": DummyStrategy,
        "DatasetH": DummyDataset,
        "DataHandlerLP": mock.Mock(),
        "init_instance_by_config": mock.Mock(return_value="handler"),
        "R": DummyR(sink),
        "Task": DummyTask,
    }


class WorkflowTests(unittest.TestCase):
    def test_parse_args(self):
        parsed = workflow.parse_args(
            [
                "--market",
                "test_market",
                "--start",
                "2022-01-01",
                "--end",
                "2022-12-31",
                "--freq",
                "day",
                "--horizon",
                "10",
                "--topk",
                "5",
                "--n-drop",
                "2",
            ]
        )
        self.assertEqual(parsed.market, "test_market")
        self.assertEqual(parsed.start, "2022-01-01")
        self.assertEqual(parsed.end, "2022-12-31")
        self.assertEqual(parsed.freq, "day")
        self.assertEqual(parsed.horizon, 10)
        self.assertEqual(parsed.topk, 5)
        self.assertEqual(parsed.n_drop, 2)

    def test_run_experiment_with_stubs(self):
        sink = {}
        deps = build_stub_dependencies(sink)
        args = workflow.WorkflowArgs(
            market="csi300",
            start="2020-01-01",
            end="2020-12-31",
            freq="day",
            horizon=20,
            topk=30,
            n_drop=10,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_root = Path(tmpdir)
            with mock.patch.object(workflow, "EXP_ROOT", exp_root), \
                mock.patch.object(workflow, "ensure_directories", autospec=True), \
                mock.patch.object(workflow, "QLIB_PROVIDER_URI", exp_root / "qlib_data"), \
                mock.patch.object(workflow, "require_dependencies", return_value=deps):
                workflow.run_experiment(args)

        self.assertIn("params", sink)
        self.assertIn("metrics", sink)
        self.assertIn("risk_analysis", sink)


if __name__ == "__main__":
    unittest.main()
