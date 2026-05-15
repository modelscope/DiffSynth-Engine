import importlib
import sys
import unittest
from unittest.mock import patch, MagicMock

from diffsynth_engine.utils.import_utils import is_npu_available


class TestIsNpuAvailable(unittest.TestCase):
    """Test NPU detection with mindiesd and torch_npu fallback paths."""

    def setUp(self):
        self._orig_modules = dict(sys.modules)

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self._orig_modules)
        importlib.invalidate_caches()

    @patch("importlib.util.find_spec", return_value=None)
    def test_no_mindiesd_no_torch_npu(self, _mock):
        self.assertFalse(is_npu_available())

    @patch("importlib.util.find_spec")
    def test_mindiesd_unavailable_torch_npu_available(self, mock_find_spec):
        """mindiesd absent, torch_npu available → True via manual fallback."""
        def find_spec_side_effect(name):
            if name == "mindiesd":
                return None
            if name == "torch_npu":
                return MagicMock()
            return None

        mock_find_spec.side_effect = find_spec_side_effect

        sys.modules["torch_npu"] = MagicMock()
        import torch

        class FakeNpu:
            device_count = MagicMock(return_value=1)
            is_available = MagicMock(return_value=True)

        torch.npu = FakeNpu()

        try:
            self.assertTrue(is_npu_available())
        finally:
            del torch.npu

    @patch("importlib.util.find_spec", return_value=None)
    def test_mindiesd_unavailable_torch_npu_unavailable(self, _mock):
        self.assertFalse(is_npu_available())

    @patch("importlib.util.find_spec")
    def test_mindiesd_available_returns_true(self, mock_find_spec):
        """mindiesd present and reports NPU available → True."""
        mock_find_spec.return_value = MagicMock()

        fake_mindiesd = MagicMock()
        fake_mindiesd.utils = MagicMock()
        fake_mindiesd.utils.is_npu_available.return_value = True
        sys.modules["mindiesd"] = fake_mindiesd
        sys.modules["mindiesd.utils"] = fake_mindiesd.utils

        try:
            self.assertTrue(is_npu_available())
        finally:
            sys.modules.pop("mindiesd", None)
            sys.modules.pop("mindiesd.utils", None)

    @patch("importlib.util.find_spec")
    def test_mindiesd_available_returns_false(self, mock_find_spec):
        """mindiesd present but reports NPU unavailable → False."""
        mock_find_spec.return_value = MagicMock()

        fake_mindiesd = MagicMock()
        fake_mindiesd.utils = MagicMock()
        fake_mindiesd.utils.is_npu_available.return_value = False
        sys.modules["mindiesd"] = fake_mindiesd
        sys.modules["mindiesd.utils"] = fake_mindiesd.utils

        try:
            self.assertFalse(is_npu_available())
        finally:
            sys.modules.pop("mindiesd", None)
            sys.modules.pop("mindiesd.utils", None)

    @patch("importlib.util.find_spec")
    def test_mindiesd_import_error_falls_back_to_torch_npu(self, mock_find_spec):
        """mindiesd spec found but is_npu_available raises ImportError → fallback."""
        def find_spec_side_effect(name):
            if name == "mindiesd":
                return MagicMock()
            if name == "torch_npu":
                return MagicMock()
            return None

        mock_find_spec.side_effect = find_spec_side_effect

        fake_mindiesd = MagicMock()

        class RaisingFrom:
            def __getattr__(self, name):
                raise ImportError("No module")

        fake_mindiesd.utils = RaisingFrom()
        sys.modules["mindiesd"] = fake_mindiesd
        sys.modules["mindiesd.utils"] = fake_mindiesd.utils

        sys.modules["torch_npu"] = MagicMock()
        import torch

        class FakeNpu:
            device_count = MagicMock(return_value=1)
            is_available = MagicMock(return_value=True)

        torch.npu = FakeNpu()

        try:
            self.assertTrue(is_npu_available())
        finally:
            sys.modules.pop("mindiesd", None)
            sys.modules.pop("mindiesd.utils", None)
            del torch.npu

    @patch("importlib.util.find_spec")
    def test_mindiesd_no_attribute_falls_back(self, mock_find_spec):
        """mindiesd has no is_npu_available attribute → fallback."""
        def find_spec_side_effect(name):
            if name == "mindiesd":
                return MagicMock()
            if name == "torch_npu":
                return MagicMock()
            return None

        mock_find_spec.side_effect = find_spec_side_effect

        fake_mindiesd = MagicMock()
        fake_mindiesd.utils = MagicMock(spec=[])  # no is_npu_available
        sys.modules["mindiesd"] = fake_mindiesd
        sys.modules["mindiesd.utils"] = fake_mindiesd.utils

        sys.modules["torch_npu"] = MagicMock()
        import torch

        class FakeNpu:
            device_count = MagicMock(return_value=1)
            is_available = MagicMock(return_value=True)

        torch.npu = FakeNpu()

        try:
            self.assertTrue(is_npu_available())
        finally:
            sys.modules.pop("mindiesd", None)
            sys.modules.pop("mindiesd.utils", None)
            del torch.npu

    @patch("importlib.util.find_spec")
    def test_torch_npu_runtime_error_returns_false(self, mock_find_spec):
        """torch_npu raises RuntimeError during detection → False."""
        def find_spec_side_effect(name):
            if name == "mindiesd":
                return None
            if name == "torch_npu":
                return MagicMock()
            return None

        mock_find_spec.side_effect = find_spec_side_effect

        sys.modules["torch_npu"] = MagicMock()
        import torch

        class FakeNpu:
            device_count = MagicMock(side_effect=RuntimeError("NPU init failed"))

        torch.npu = FakeNpu()

        try:
            self.assertFalse(is_npu_available())
        finally:
            del torch.npu

    def test_smoke_non_npu_system(self):
        """is_npu_available returns bool on non-NPU systems."""
        result = is_npu_available()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()