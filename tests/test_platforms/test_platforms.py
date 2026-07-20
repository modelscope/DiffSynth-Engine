import types
import unittest
from unittest import mock

import torch

from diffsynth_engine.platforms import (
    get_device_type,
    resolve_platform,
)
from diffsynth_engine.platforms import ascend
from diffsynth_engine.utils import platform as compatibility_platform


class TestPlatformResolution(unittest.TestCase):
    def test_resolution_only_uses_explicit_device(self):
        with mock.patch.object(ascend.AscendPlatform, "is_available", return_value=True):
            self.assertIn(resolve_platform("cuda").name, {"cuda", "rocm"})
            self.assertEqual(resolve_platform("npu:0").name, "ascend")
        self.assertEqual(get_device_type(torch.device("cpu")), "cpu")

    def test_auto_device_is_not_silently_resolved(self):
        with self.assertRaisesRegex(ValueError, "Unsupported device type 'auto'"):
            resolve_platform("auto")

    def test_resolution_does_not_import_torch_npu(self):
        with mock.patch.object(ascend, "_import_torch_npu") as import_torch_npu:
            self.assertEqual(resolve_platform("npu:0").name, "ascend")
        import_torch_npu.assert_not_called()

    def test_legacy_cache_cleanup_does_not_implicitly_touch_npu(self):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch.backends.mps, "is_available", return_value=False),
            mock.patch.object(ascend.AscendPlatform, "empty_cache") as npu_empty_cache,
        ):
            compatibility_platform.empty_cache()
        npu_empty_cache.assert_not_called()


class TestAscendCapabilities(unittest.TestCase):
    def tearDown(self):
        ascend.reset_ascend_capability_cache()

    @staticmethod
    def _fake_torch_npu():
        return types.SimpleNamespace(
            npu=types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda index: "Ascend"),
            npu_dynamic_mx_quant=lambda *args, **kwargs: None,
            npu_quant_matmul=lambda *args, **kwargs: None,
            npu_dynamic_block_quant=lambda *args, **kwargs: None,
            npu_fused_infer_attention_score_v2=lambda *args, **kwargs: None,
            float4_e2m1fn_x2=object(),
            float8_e4m3fn=object(),
        )

    def test_capability_probe_checks_apis_and_operators(self):
        class QuantAlgorithm:
            W8A8_MXFP8 = object()
            W4A4_MXFP4_DYNAMIC = object()
            FP8_DYNAMIC = object()

        modules = {
            "mindiesd.layers.flash_attn.attention_forward": types.SimpleNamespace(
                attention_forward=lambda *args, **kwargs: None
            ),
            "mindiesd.compilation": types.SimpleNamespace(MindieSDBackend=lambda: object()),
            "mindiesd.quantization": types.SimpleNamespace(
                quantize=lambda *args, **kwargs: None,
                OnlineQuantConfig=object,
            ),
            "mindiesd.quantization.mode": types.SimpleNamespace(QuantAlgorithm=QuantAlgorithm),
            "mindiesd.quantization.layer": types.SimpleNamespace(FP8RotateQuantFA=object),
        }
        operation_probes = {name: mock.Mock(return_value=True) for name in ascend._OPERATION_PROBES}

        ascend.reset_ascend_capability_cache()
        with (
            mock.patch.object(ascend, "_import_torch_npu", return_value=self._fake_torch_npu()),
            mock.patch.object(ascend, "_probe_npu_runtime", return_value=True),
            mock.patch.object(ascend, "_import_mindie_sd", return_value=object()),
            mock.patch.object(ascend.importlib, "import_module", side_effect=modules.__getitem__),
            mock.patch.dict(ascend._OPERATION_PROBES, operation_probes),
        ):
            capabilities = ascend.probe_ascend_capabilities()
            cached_capabilities = ascend.probe_ascend_capabilities()

        self.assertTrue(capabilities.device)
        self.assertTrue(capabilities.mindie_attention)
        self.assertTrue(capabilities.mindie_compile)
        self.assertTrue(capabilities.mindie_mxfp8_linear)
        self.assertTrue(capabilities.mindie_w4a4_linear)
        self.assertTrue(capabilities.mindie_fp8_attention)
        self.assertEqual(capabilities, cached_capabilities)
        for operation_probe in operation_probes.values():
            operation_probe.assert_called_once_with()

    def test_failed_operator_probe_disables_only_that_feature(self):
        ascend.reset_ascend_capability_cache()
        with (
            mock.patch.object(ascend, "_probe_ascend_device", return_value=True),
            mock.patch.object(ascend, "_probe_mindie_installation", return_value=True),
            mock.patch.object(ascend, "_feature_api_available", return_value=True),
            mock.patch.dict(
                ascend._OPERATION_PROBES,
                {"mindie_attention": mock.Mock(side_effect=RuntimeError("operator unavailable"))},
            ),
        ):
            self.assertFalse(ascend.probe_ascend_feature("mindie_attention"))

    def test_missing_torch_npu_reports_no_device(self):
        ascend.reset_ascend_capability_cache()
        with mock.patch.object(ascend, "_import_torch_npu", side_effect=RuntimeError("missing")):
            capabilities = ascend.probe_ascend_capabilities()
        self.assertFalse(capabilities.device)

    def test_mindie_missing_keeps_basic_npu_available(self):
        ascend.reset_ascend_capability_cache()
        with (
            mock.patch.object(ascend, "_import_torch_npu", return_value=self._fake_torch_npu()),
            mock.patch.object(ascend, "_probe_npu_runtime", return_value=True),
            mock.patch.object(ascend, "_import_mindie_sd", side_effect=RuntimeError("missing")),
        ):
            capabilities = ascend.probe_ascend_capabilities()

        self.assertTrue(capabilities.device)
        self.assertFalse(capabilities.mindie)
        self.assertFalse(capabilities.mindie_attention)

    def test_incompatible_mindie_api_disables_feature_without_breaking_npu(self):
        ascend.reset_ascend_capability_cache()
        with (
            mock.patch.object(ascend, "_import_torch_npu", return_value=self._fake_torch_npu()),
            mock.patch.object(ascend, "_probe_npu_runtime", return_value=True),
            mock.patch.object(ascend, "_import_mindie_sd", return_value=object()),
            mock.patch.object(ascend.importlib, "import_module", side_effect=ValueError("incompatible API")),
        ):
            capabilities = ascend.probe_ascend_capabilities()

        self.assertTrue(capabilities.device)
        self.assertTrue(capabilities.mindie)
        self.assertFalse(capabilities.mindie_attention)
        self.assertFalse(capabilities.mindie_compile)
        self.assertFalse(capabilities.mindie_mxfp8_linear)

    def test_runtime_probe_failure_overrides_is_available(self):
        ascend.reset_ascend_capability_cache()
        with (
            mock.patch.object(ascend, "_import_torch_npu", return_value=self._fake_torch_npu()),
            mock.patch.object(ascend, "_probe_npu_runtime", return_value=False),
        ):
            capabilities = ascend.probe_ascend_capabilities()

        self.assertFalse(capabilities.device)


if __name__ == "__main__":
    unittest.main()
