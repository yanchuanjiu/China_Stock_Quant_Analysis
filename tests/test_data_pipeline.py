import unittest
from unittest import mock

from qlib_cn_lowfreq import data_pipeline


class DataPipelineTests(unittest.TestCase):
    def test_parse_args_defaults(self):
        parsed = data_pipeline.parse_args([])
        self.assertEqual(parsed.start, data_pipeline.DEFAULT_DATE_RANGE.start)
        self.assertEqual(parsed.end, data_pipeline.DEFAULT_DATE_RANGE.end)
        self.assertEqual(parsed.provider_uri, str(data_pipeline.QLIB_PROVIDER_URI))

    def test_require_dependencies_missing(self):
        with mock.patch("importlib.import_module", side_effect=ImportError):
            with self.assertRaises(data_pipeline.MissingDependencyError):
                data_pipeline.require_dependencies()


if __name__ == "__main__":
    unittest.main()
