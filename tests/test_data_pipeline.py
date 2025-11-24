import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from qlib_cn_lowfreq import data_pipeline


def _make_hist(symbol_code: str, start_value: float = 1.0) -> pd.DataFrame:
    base_rows = [
        {
            "日期": "2023-01-02",
            "开盘": start_value,
            "收盘": start_value + 1,
            "最高": start_value + 2,
            "最低": start_value - 1,
            "成交量": 1000,
            "成交额": 2000,
        },
        {
            "日期": "2023-01-03",
            "开盘": start_value + 3,
            "收盘": start_value + 4,
            "最高": start_value + 5,
            "最低": start_value + 2,
            "成交量": 1100,
            "成交额": 2100,
        },
    ]
    df = pd.DataFrame(base_rows)
    df["symbol"] = symbol_code
    return df


class FakeAk:
    def __init__(self, listings: pd.DataFrame, history: dict[str, pd.DataFrame]):
        self._listings = listings
        self._history = history

    def stock_zh_a_spot_em(self):
        return self._listings.copy()

    def stock_zh_a_hist_ths(self, symbol, **_kwargs):
        return self._history.get(symbol, pd.DataFrame()).copy()

    def stock_zh_a_hist(self, symbol, **_kwargs):
        return self._history.get(symbol, pd.DataFrame()).copy()


class StubRecorder:
    def __init__(self):
        self.logged_params = None
        self.logged_artifacts: list[str] = []

    def log_params(self, params):
        self.logged_params = params

    def log_artifact(self, artifact_path: str):
        self.logged_artifacts.append(artifact_path)


class StubR:
    def __init__(self, recorder: StubRecorder):
        self.recorder = recorder
        self.calls = []

    def start(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        recorder = self.recorder

        class _Ctx:
            def __enter__(self_inner):
                return recorder

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


class DataPipelineTests(unittest.TestCase):
    def test_find_local_dump_script_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.assertIsNone(data_pipeline._find_local_dump_script(base_dir=tmp_path))

    def test_find_local_dump_script_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            scripts_dir = tmp_path / "qlib" / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            candidate = scripts_dir / "dump_bin.py"
            candidate.write_text("# mock\n")
            self.assertEqual(data_pipeline._find_local_dump_script(base_dir=tmp_path), candidate)

    def test_start_recorder_legacy_kwargs(self):
        class LegacyR:
            def __init__(self):
                self.calls = []

            def start(self, **kwargs):
                self.calls.append(kwargs)
                if "exp_name" in kwargs:
                    raise TypeError("unsupported")

                class _Ctx:
                    def __enter__(self_inner):
                        return SimpleNamespace(log_params=lambda *_: None, log_artifact=lambda *_: None)

                    def __exit__(self_inner, exc_type, exc, tb):
                        return False

                return _Ctx()

        legacy_r = LegacyR()
        ctx = data_pipeline._start_recorder(legacy_r, uri="test")
        self.assertIsNotNone(ctx)
        self.assertEqual(len(legacy_r.calls), 2)  # exp_name attempt + fallback

    def test_start_recorder_minimal_interface(self):
        class MinimalR:
            def __init__(self):
                self.calls = []

            def start(self, **kwargs):
                self.calls.append(kwargs)
                if "exp_name" in kwargs or "experiment_name" in kwargs:
                    raise TypeError("unsupported")

                class _Ctx:
                    def __enter__(self_inner):
                        return SimpleNamespace(log_params=lambda *_: None, log_artifact=lambda *_: None)

                    def __exit__(self_inner, exc_type, exc, tb):
                        return False

                return _Ctx()

        minimal_r = MinimalR()
        ctx = data_pipeline._start_recorder(minimal_r, uri="test")
        self.assertIsNotNone(ctx)
        self.assertEqual(len(minimal_r.calls), 3)

    def test_coerce_recorder_prefers_active(self):
        active = SimpleNamespace(log_params=lambda *_: None)
        wrapper = SimpleNamespace(active_recorder=active)
        self.assertIs(data_pipeline._coerce_recorder(wrapper), active)

    def test_coerce_recorder_raises_on_invalid_object(self):
        with self.assertRaises(RuntimeError):
            data_pipeline._coerce_recorder(object())

    def test_parse_args_defaults(self):
        parsed = data_pipeline.parse_args([])
        self.assertEqual(parsed.start, data_pipeline.DEFAULT_DATE_RANGE.start)
        self.assertEqual(parsed.end, data_pipeline.DEFAULT_DATE_RANGE.end)
        self.assertEqual(parsed.provider_uri, str(data_pipeline.QLIB_PROVIDER_URI))

    def test_require_dependencies_missing(self):
        with mock.patch("importlib.import_module", side_effect=ImportError):
            with self.assertRaises(data_pipeline.MissingDependencyError):
                data_pipeline.require_dependencies()

    def test_require_dependencies_success(self):
        ak_mod = object()
        qlib_mod = SimpleNamespace(init=mock.Mock())
        utils_mod = SimpleNamespace(convert_index_format=lambda df: df, dump_bin=lambda *a, **k: None)
        workflow_mod = SimpleNamespace(R=object())

        def fake_import(name):
            mapping = {
                "akshare": ak_mod,
                "pandas": pd,
                "qlib": qlib_mod,
                "qlib.data.dataset.utils": utils_mod,
                "qlib.workflow": workflow_mod,
            }
            if name not in mapping:
                return object()
            return mapping[name]

        with mock.patch("importlib.import_module", side_effect=fake_import):
            deps = data_pipeline.require_dependencies()

        self.assertIs(deps["ak"], ak_mod)
        self.assertIs(deps["qlib"], qlib_mod)
        self.assertIs(deps["convert_index_format"], utils_mod.convert_index_format)
        self.assertIs(deps["dump_bin"], utils_mod.dump_bin)
        self.assertIs(deps["R"], workflow_mod.R)
        self.assertIsNone(deps["DumpDataAll"])
        self.assertIsNone(deps["dump_bin_script"])

    def test_require_dependencies_fallback_dump_data_all(self):
        ak_mod = object()
        qlib_mod = SimpleNamespace(init=mock.Mock())
        utils_mod = SimpleNamespace(convert_index_format=lambda df: df)
        dump_module = SimpleNamespace(DumpDataAll=object(), __file__="/tmp/dump_bin.py")
        workflow_mod = SimpleNamespace(R=object())

        def fake_import(name):
            mapping = {
                "akshare": ak_mod,
                "pandas": pd,
                "qlib": qlib_mod,
                "qlib.data.dataset.utils": utils_mod,
                "qlib.workflow": workflow_mod,
                "qlib.scripts.dump_bin": dump_module,
            }
            if name not in mapping:
                raise ImportError(name)
            return mapping[name]

        with mock.patch("importlib.import_module", side_effect=fake_import):
            deps = data_pipeline.require_dependencies()

        self.assertIsNone(deps["dump_bin"])
        self.assertIs(deps["DumpDataAll"], dump_module.DumpDataAll)
        self.assertEqual(deps["dump_bin_script"], Path("/tmp/dump_bin.py"))

    def test_require_dependencies_without_dump_support(self):
        ak_mod = object()
        qlib_mod = SimpleNamespace(init=mock.Mock())
        utils_mod = SimpleNamespace(convert_index_format=lambda df: df)
        workflow_mod = SimpleNamespace(R=object())

        def fake_import(name):
            mapping = {
                "akshare": ak_mod,
                "pandas": pd,
                "qlib": qlib_mod,
                "qlib.data.dataset.utils": utils_mod,
                "qlib.workflow": workflow_mod,
            }
            if name in mapping:
                return mapping[name]
            if name == "qlib.scripts.dump_bin":
                raise ImportError("missing")
            raise ImportError(name)

        with mock.patch("importlib.import_module", side_effect=fake_import), \
            mock.patch.object(data_pipeline, "_find_local_dump_script", return_value=None):
            deps = data_pipeline.require_dependencies()

        self.assertIsNone(deps["dump_bin"])
        self.assertIsNone(deps["DumpDataAll"])
        self.assertIsNone(deps["dump_bin_script"])

    def test_require_dependencies_loads_local_dump_script(self):
        ak_mod = object()
        qlib_mod = SimpleNamespace(init=mock.Mock())
        utils_mod = SimpleNamespace(convert_index_format=lambda df: df)
        workflow_mod = SimpleNamespace(R=object())
        local_dump_path = Path("/repo/qlib/scripts/dump_bin.py")

        def fake_import(name):
            mapping = {
                "akshare": ak_mod,
                "pandas": pd,
                "qlib": qlib_mod,
                "qlib.data.dataset.utils": utils_mod,
                "qlib.workflow": workflow_mod,
            }
            if name in mapping:
                return mapping[name]
            if name == "qlib.scripts.dump_bin":
                raise ImportError("missing remote script")
            raise ImportError(name)

        with mock.patch("importlib.import_module", side_effect=fake_import), \
            mock.patch.object(data_pipeline, "_find_local_dump_script", return_value=local_dump_path):
            deps = data_pipeline.require_dependencies()

        self.assertIsNone(deps["dump_bin"])
        self.assertIsNone(deps["DumpDataAll"])
        self.assertEqual(deps["dump_bin_script"], local_dump_path)

    def test_list_instruments_prefers_user_symbols(self):
        with mock.patch.object(data_pipeline, "require_dependencies", return_value={"ak": None, "pd": pd}):
            fetcher = data_pipeline.TongHuaShunFetcher(start="2023-01-01", end="2023-01-02", symbols=["000001.SZ"])
        self.assertEqual(fetcher.list_instruments(), ["000001.SZ"])

    def test_list_instruments_from_akshare(self):
        listings = pd.DataFrame({"代码": ["000001", "600000"]})
        fake_ak = FakeAk(listings=listings, history={})
        with mock.patch.object(
            data_pipeline,
            "require_dependencies",
            return_value={"ak": fake_ak, "pd": pd},
        ):
            fetcher = data_pipeline.TongHuaShunFetcher(start="2023-01-01", end="2023-01-02")
            symbols = fetcher.list_instruments()
        self.assertEqual(symbols, ["000001.SZ", "600000.SH"])

    def test_fetch_symbol_formats_dataframe(self):
        listings = pd.DataFrame({"代码": ["000001"]})
        history = {"000001": _make_hist("000001")}
        fake_ak = FakeAk(listings=listings, history=history)
        with mock.patch.object(
            data_pipeline,
            "require_dependencies",
            return_value={"ak": fake_ak, "pd": pd},
        ):
            fetcher = data_pipeline.TongHuaShunFetcher(start="2023-01-01", end="2023-01-31", symbols=["000001.SZ"])
            result = fetcher.fetch_symbol("000001.SZ")

        self.assertEqual(list(result.columns), ["open", "high", "low", "close", "volume", "amount", "symbol"])
        self.assertTrue((result["symbol"] == "000001.SZ").all())
        self.assertEqual(result.index.name, "date")

    def test_fetch_symbol_fallback_hist_api(self):
        listings = pd.DataFrame({"代码": ["000001"]})
        history = {"000001": _make_hist("000001")}
        fake_ak = FakeAk(listings=listings, history=history)
        fake_ak.stock_zh_a_hist_ths = None

        with mock.patch.object(
            data_pipeline,
            "require_dependencies",
            return_value={"ak": fake_ak, "pd": pd},
        ):
            fetcher = data_pipeline.TongHuaShunFetcher(start="2023-01-01", end="2023-01-31", symbols=["000001.SZ"])
            result = fetcher.fetch_symbol("000001.SZ")

        self.assertFalse(result.empty)
        self.assertTrue((result["symbol"] == "000001.SZ").all())

    def test_fetch_symbol_no_hist_api_raises(self):
        listings = pd.DataFrame({"代码": ["000001"]})
        history = {}
        fake_ak = FakeAk(listings=listings, history=history)
        fake_ak.stock_zh_a_hist_ths = None
        fake_ak.stock_zh_a_hist = None

        with mock.patch.object(
            data_pipeline,
            "require_dependencies",
            return_value={"ak": fake_ak, "pd": pd},
        ):
            fetcher = data_pipeline.TongHuaShunFetcher(start="2023-01-01", end="2023-01-31", symbols=["000001.SZ"])
            with self.assertRaises(data_pipeline.MissingDependencyError):
                fetcher.fetch_symbol("000001.SZ")

    def test_fetch_symbol_empty_dataframe_raises(self):
        listings = pd.DataFrame({"代码": ["000001"]})
        fake_ak = FakeAk(listings=listings, history={"000001": pd.DataFrame()})
        with mock.patch.object(
            data_pipeline,
            "require_dependencies",
            return_value={"ak": fake_ak, "pd": pd},
        ):
            fetcher = data_pipeline.TongHuaShunFetcher(start="2023-01-01", end="2023-01-31", symbols=["000001.SZ"])
            with self.assertRaises(ValueError):
                fetcher.fetch_symbol("000001.SZ")

    def test_fetch_symbol_missing_date_column_raises(self):
        listings = pd.DataFrame({"代码": ["000001"]})
        faulty_df = pd.DataFrame(
            {
                "开盘": [1.0],
                "收盘": [1.1],
                "最高": [1.2],
                "最低": [0.9],
                "成交量": [1000],
                "成交额": [2000],
            }
        )
        fake_ak = FakeAk(listings=listings, history={"000001": faulty_df})

        with mock.patch.object(
            data_pipeline,
            "require_dependencies",
            return_value={"ak": fake_ak, "pd": pd},
        ):
            fetcher = data_pipeline.TongHuaShunFetcher(start="2023-01-01", end="2023-01-31", symbols=["000001.SZ"])
            with self.assertRaises(ValueError):
                fetcher.fetch_symbol("000001.SZ")

    def test_fetch_all_concatenates_all_symbols(self):
        listings = pd.DataFrame({"代码": ["000001", "600000"]})
        history = {
            "000001": _make_hist("000001", start_value=1.0),
            "600000": _make_hist("600000", start_value=10.0),
        }
        fake_ak = FakeAk(listings=listings, history=history)
        with mock.patch.object(
            data_pipeline,
            "require_dependencies",
            return_value={"ak": fake_ak, "pd": pd},
        ):
            fetcher = data_pipeline.TongHuaShunFetcher(
                start="2023-01-01",
                end="2023-01-31",
                symbols=["000001.SZ", "600000.SH"],
            )
            result = fetcher.fetch_all()

        self.assertEqual(result["symbol"].nunique(), 2)
        self.assertIn("symbol", result.columns)

    def test_dump_to_qlib_store_invokes_qlib_tools(self):
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100],
                "amount": [200],
                "symbol": ["000001.SZ"],
            },
            index=pd.to_datetime(["2023-01-02"]),
        )
        df.index.name = "date"

        qlib_stub = SimpleNamespace(init=mock.Mock())
        convert_mock = mock.Mock(side_effect=lambda x: x)
        dump_mock = mock.Mock()

        deps = {
            "qlib": qlib_stub,
            "convert_index_format": convert_mock,
            "dump_bin": dump_mock,
            "DumpDataAll": None,
            "pd": pd,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = Path(tmpdir) / "qlib_data"
            with mock.patch.object(data_pipeline, "require_dependencies", return_value=deps):
                data_pipeline.dump_to_qlib_store(df, provider_uri=provider)

        qlib_stub.init.assert_called_once()
        convert_mock.assert_called_once()
        dump_mock.assert_called_once()

    def test_dump_to_qlib_store_handles_mirror(self):
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100],
                "amount": [200],
                "symbol": ["000001.SZ"],
            },
            index=pd.to_datetime(["2023-01-02"]),
        )
        df.index.name = "date"

        deps = {
            "qlib": SimpleNamespace(init=mock.Mock()),
            "convert_index_format": lambda x: x,
            "dump_bin": lambda *args, **kwargs: None,
            "DumpDataAll": None,
            "dump_bin_script": None,
            "pd": pd,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = Path(tmpdir) / "qlib_data"
            mirror = Path(tmpdir) / "mirror"
            with mock.patch.object(data_pipeline, "require_dependencies", return_value=deps), \
                mock.patch("shutil.copytree") as copy_mock:
                data_pipeline.dump_to_qlib_store(df, provider_uri=provider, mirror_to=mirror)
        copy_mock.assert_called_once_with(provider, mirror, dirs_exist_ok=True)

    def test_dump_to_qlib_store_rejects_empty_dataframe(self):
        empty_df = pd.DataFrame()
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = Path(tmpdir) / "qlib_data"
            with self.assertRaises(ValueError):
                data_pipeline.dump_to_qlib_store(empty_df, provider_uri=provider)

    def test_dump_to_qlib_store_uses_dump_data_all_when_missing_dump_bin(self):
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100],
                "amount": [200],
                "symbol": ["000001.SZ"],
            },
            index=pd.to_datetime(["2023-01-02"]),
        )
        df.index.name = "date"

        class StubDump:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                StubDump.instances.append(self)

            def dump(self):
                self.dump_called = True

        deps = {
            "qlib": SimpleNamespace(init=mock.Mock()),
            "convert_index_format": lambda x: x,
            "dump_bin": None,
            "DumpDataAll": StubDump,
            "dump_bin_script": None,
            "pd": pd,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = Path(tmpdir) / "qlib_data"
            with mock.patch.object(data_pipeline, "require_dependencies", return_value=deps):
                data_pipeline.dump_to_qlib_store(df, provider_uri=provider)

        self.assertTrue(StubDump.instances)
        self.assertTrue(getattr(StubDump.instances[0], "dump_called", False))

    def test_dump_to_qlib_store_raises_when_no_dump_support(self):
        df = pd.DataFrame({"symbol": ["000001.SZ"]}, index=pd.to_datetime(["2023-01-02"]))
        df.index.name = "date"

        deps = {
            "qlib": SimpleNamespace(init=mock.Mock()),
            "convert_index_format": lambda x: x,
            "dump_bin": None,
            "DumpDataAll": None,
            "dump_bin_script": None,
            "pd": pd,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = Path(tmpdir) / "qlib_data"
            with mock.patch.object(data_pipeline, "require_dependencies", return_value=deps):
                with self.assertRaises(RuntimeError):
                    data_pipeline.dump_to_qlib_store(df, provider_uri=provider)

    def test_dump_to_qlib_store_invokes_cli_when_only_script_available(self):
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100],
                "amount": [200],
                "symbol": ["000001.SZ"],
            },
            index=pd.to_datetime(["2023-01-02"]),
        )
        df.index.name = "date"

        deps = {
            "qlib": SimpleNamespace(init=mock.Mock()),
            "convert_index_format": lambda x: x,
            "dump_bin": None,
            "DumpDataAll": None,
            "dump_bin_script": Path("/tmp/dump_bin.py"),
            "pd": pd,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = Path(tmpdir) / "qlib_data"
            with mock.patch.object(data_pipeline, "require_dependencies", return_value=deps), \
                mock.patch("subprocess.run") as run_mock:
                data_pipeline.dump_to_qlib_store(df, provider_uri=provider)

        run_mock.assert_called_once()
        self.assertIn("/tmp/dump_bin.py", run_mock.call_args[0][0])

    def test_main_runs_end_to_end_with_stubs(self):
        sample_df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100],
                "amount": [200],
                "symbol": ["000001.SZ"],
            },
            index=pd.to_datetime(["2023-01-02"]),
        )
        sample_df.index.name = "date"

        recorder = StubRecorder()
        stub_r = StubR(recorder)
        fetcher_instance = mock.Mock()
        fetcher_instance.fetch_all.return_value = sample_df

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = Path(tmpdir) / "qlib_data"
            akshare_dir = Path(tmpdir) / "akshare"
            mirror_dir = Path(tmpdir) / "mirror"

            with mock.patch.object(data_pipeline, "AKSHARE_DATA_DIR", akshare_dir), \
                mock.patch.object(data_pipeline, "DEFAULT_YAHOO_PROVIDER", mirror_dir), \
                mock.patch.object(data_pipeline, "TongHuaShunFetcher", return_value=fetcher_instance), \
                mock.patch.object(data_pipeline, "dump_to_qlib_store") as dump_mock, \
                mock.patch.object(data_pipeline, "ensure_directories") as ensure_dirs, \
                mock.patch.object(data_pipeline, "require_dependencies", return_value={"R": stub_r}):
                data_pipeline.main(
                    [
                        "--start",
                        "2023-01-01",
                        "--end",
                        "2023-01-31",
                        "--symbols",
                        "000001.SZ",
                        "--provider-uri",
                        str(provider),
                        "--mirror-yahoo",
                    ]
                )

        ensure_dirs.assert_called_once()
        fetcher_instance.fetch_all.assert_called_once()
        dump_mock.assert_called_once()
        self.assertEqual(recorder.logged_params["start"], "2023-01-01")
        resolved_artifacts = {Path(p).resolve() for p in recorder.logged_artifacts}
        self.assertIn(provider.resolve(), resolved_artifacts)
        self.assertEqual(stub_r.calls[0]["kwargs"]["uri"], str((akshare_dir / "mlruns").resolve()))


if __name__ == "__main__":
    unittest.main()
