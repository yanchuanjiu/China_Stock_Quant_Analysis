import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import qlib_cn_lowfreq.config as config


class ConfigTests(unittest.TestCase):
    def test_ensure_directories_creates_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            ak_dir = base / "akshare_ths"
            qlib_dir = ak_dir / "qlib_data"
            exp_dir = base / "experiments"

            with patch.object(config, "AKSHARE_DATA_DIR", ak_dir), \
                patch.object(config, "QLIB_PROVIDER_URI", qlib_dir), \
                patch.object(config, "EXP_ROOT", exp_dir):
                config.ensure_directories()

            self.assertTrue(ak_dir.exists())
            self.assertTrue(qlib_dir.exists())
            self.assertTrue(exp_dir.exists())
