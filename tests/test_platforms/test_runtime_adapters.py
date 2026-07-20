import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

from diffsynth_engine.configs import QuantizationConfig
from diffsynth_engine.platforms import PlatformCapabilities
from diffsynth_engine.platforms import ascend_qwen
from diffsynth_engine.platforms.ascend_qwen import AscendQwenImageRuntimeAdapter
from diffsynth_engine.platforms.qwen import CudaQwenImageRuntimeAdapter
from diffsynth_engine.platforms.runtime import DefaultRuntimeAdapter


class FakePlatform:
    name = "ascend"
    current_capabilities = PlatformCapabilities(
        device=True,
        mindie=True,
        mindie_attention=True,
        mindie_compile=True,
        mindie_mxfp8_linear=True,
        mindie_w4a4_linear=True,
        mindie_fp8_attention=True,
    )

    @classmethod
    def capabilities(cls):
        return cls.current_capabilities

    @classmethod
    def supports(cls, capability):
        return bool(getattr(cls.current_capabilities, capability))

    @classmethod
    def compile_kwargs(cls):
        return {"backend": "mindie"}


def make_config(**overrides):
    values = {
        "device": "npu:0",
        "parallelism": 1,
        "dit_attn_impl": types.SimpleNamespace(value="auto"),
        "quantization": None,
        "use_fp8_linear": False,
        "use_nunchaku": False,
        "use_torch_compile": False,
        "offload_mode": None,
        "model_dtype": torch.bfloat16,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


class TestAscendQwenRuntimeAdapter(unittest.TestCase):
    def setUp(self):
        FakePlatform.current_capabilities = PlatformCapabilities(
            device=True,
            mindie=True,
            mindie_attention=True,
            mindie_compile=True,
            mindie_mxfp8_linear=True,
            mindie_w4a4_linear=True,
            mindie_fp8_attention=True,
        )
        self.adapter = AscendQwenImageRuntimeAdapter(FakePlatform, "qwen_image")

    def test_rejects_first_release_incompatibilities(self):
        invalid = (
            ({"parallelism": 2}, "Multi-NPU"),
            ({"use_nunchaku": True}, "cannot run on Ascend"),
            ({"use_torch_compile": True, "offload_mode": "cpu_offload"}, "compilation cannot be combined"),
            (
                {"quantization": QuantizationConfig(linear="fp8"), "offload_mode": "cpu_offload"},
                "quantization cannot be combined",
            ),
            (
                {"quantization": QuantizationConfig(linear="fp8"), "use_torch_compile": True},
                "quantization and compilation",
            ),
        )
        for overrides, message in invalid:
            with self.subTest(overrides=overrides), self.assertRaisesRegex((ValueError, RuntimeError), message):
                self.adapter.validate_config(make_config(**overrides))

    def test_explicit_mindie_fails_before_model_loading(self):
        FakePlatform.current_capabilities = PlatformCapabilities(device=True)
        config = make_config(dit_attn_impl=types.SimpleNamespace(value="mindie"))
        with self.assertRaisesRegex(RuntimeError, "explicitly requested"):
            self.adapter.validate_config(config)

    def test_basic_bf16_allows_mindie_to_be_optional(self):
        FakePlatform.current_capabilities = PlatformCapabilities(device=True)
        self.adapter.validate_config(make_config())

    def test_quantization_capability_is_not_silently_downgraded(self):
        FakePlatform.current_capabilities = PlatformCapabilities(device=True, mindie=True)
        with self.assertRaisesRegex(RuntimeError, "MXFP8"):
            self.adapter.validate_config(make_config(quantization=QuantizationConfig(linear="fp8")))

    def test_quantization_modes_map_to_mindie_algorithms(self):
        class QuantAlgorithm:
            W8A8_MXFP8 = "mxfp8"
            W4A4_MXFP4_DYNAMIC = "mxfp4"
            W8A8_DYNAMIC = "dynamic"
            W16A16 = "bf16"
            FP8_DYNAMIC = "fp8_attention"

        records = []

        class OnlineQuantConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                records.append(self)

        quantization_module = types.SimpleNamespace(
            OnlineQuantConfig=OnlineQuantConfig,
            quantize=lambda module, **kwargs: module,
        )
        mode_module = types.SimpleNamespace(QuantAlgorithm=QuantAlgorithm)

        for linear, expected in (("fp8", "mxfp8"), ("int4", "mxfp4")):
            with self.subTest(linear=linear), mock.patch.object(
                ascend_qwen.importlib,
                "import_module",
                side_effect=lambda name: mode_module if name.endswith(".mode") else quantization_module,
            ):
                module = nn.Sequential(nn.Linear(4, 4))
                self.adapter.prepare_component(
                    "dit",
                    module,
                    make_config(quantization=QuantizationConfig(linear=linear)),
                )
            self.assertEqual(records[-1].quant_type, expected)

    def test_attention_only_quantization_keeps_linear_layers_unquantized(self):
        class QuantAlgorithm:
            W8A8_DYNAMIC = "dynamic"
            W16A16 = "bf16"
            FP8_DYNAMIC = "fp8_attention"

        record = {}

        class OnlineQuantConfig:
            def __init__(self, **kwargs):
                record.update(kwargs)

        quantization_module = types.SimpleNamespace(
            OnlineQuantConfig=OnlineQuantConfig,
            quantize=lambda module, **kwargs: module,
        )
        mode_module = types.SimpleNamespace(QuantAlgorithm=QuantAlgorithm)
        module = nn.Sequential(nn.Linear(4, 4), nn.Sequential(nn.Linear(4, 4)))
        with mock.patch.object(
            ascend_qwen.importlib,
            "import_module",
            side_effect=lambda name: mode_module if name.endswith(".mode") else quantization_module,
        ):
            self.adapter.prepare_component(
                "dit",
                module,
                make_config(quantization=QuantizationConfig(attention="fp8")),
            )

        self.assertEqual(record["fallback_layers"], {"0": "bf16", "1.0": "bf16"})
        self.assertEqual(record["fa_quant_type"], "fp8_attention")

    def test_fp8_attention_is_installed_through_processor_slot(self):
        class QuantAlgorithm:
            W8A8_DYNAMIC = "dynamic"
            W16A16 = "bf16"
            FP8_DYNAMIC = "fp8_attention"

        class AttentionLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.fa_quant = mock.Mock()
                self.processor = None

            def set_attention_processor(self, processor):
                self.processor = processor

        class OnlineQuantConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        quantization_module = types.SimpleNamespace(
            OnlineQuantConfig=OnlineQuantConfig,
            quantize=lambda module, **kwargs: module,
        )
        mode_module = types.SimpleNamespace(QuantAlgorithm=QuantAlgorithm)
        module = AttentionLayer()
        with mock.patch.object(
            ascend_qwen.importlib,
            "import_module",
            side_effect=lambda name: mode_module if name.endswith(".mode") else quantization_module,
        ):
            self.adapter.prepare_component(
                "dit",
                module,
                make_config(quantization=QuantizationConfig(attention="fp8")),
            )

        self.assertIsInstance(module.processor, ascend_qwen.MindIEFP8AttentionProcessor)

    def test_timestep_and_dynamic_weight_hooks(self):
        timestep_manager = mock.Mock()
        quantization_module = types.SimpleNamespace(TimestepManager=timestep_manager)
        self.adapter._quantization_enabled = True
        with mock.patch.object(ascend_qwen.importlib, "import_module", return_value=quantization_module):
            self.adapter.before_denoise_step(7)
        timestep_manager.set_timestep_idx.assert_called_once_with(7)

        with self.assertRaisesRegex(ValueError, "Dynamic LoRA/ControlNet"):
            self.adapter.validate_dynamic_weights(make_config(use_torch_compile=True))
        with self.assertRaisesRegex(ValueError, "Dynamic LoRA/ControlNet"):
            self.adapter.validate_dynamic_weights(
                make_config(quantization=QuantizationConfig(linear="fp8"))
            )

    def test_compile_routes_backend_to_repeated_blocks(self):
        module = mock.Mock()
        module.compile_repeated_blocks = mock.Mock()
        self.assertIs(self.adapter.compile_component("dit", module), module)
        module.compile_repeated_blocks.assert_called_once_with(backend="mindie")


class TestOtherRuntimeAdapters(unittest.TestCase):
    def test_default_compile_preserves_empty_torch_compile_kwargs(self):
        platform = types.SimpleNamespace(compile_kwargs=lambda: {})
        module = mock.Mock()
        module.compile_repeated_blocks = mock.Mock()
        adapter = DefaultRuntimeAdapter(platform, "qwen_image")
        self.assertIs(adapter.compile_component("dit", module), module)
        module.compile_repeated_blocks.assert_called_once_with()

    def test_cuda_quantization_config_uses_existing_fp8_path(self):
        adapter = CudaQwenImageRuntimeAdapter(types.SimpleNamespace(), "qwen_image")
        module = nn.Linear(4, 4)
        config = make_config(
            device="cuda",
            quantization=QuantizationConfig(backend="native", linear="fp8"),
        )
        with mock.patch("diffsynth_engine.platforms.qwen.enable_fp8_linear") as enable_fp8:
            self.assertIs(adapter.prepare_component("dit", module, config), module)
        enable_fp8.assert_called_once_with(module)

    def test_cuda_rejects_mindie_quantization(self):
        adapter = CudaQwenImageRuntimeAdapter(types.SimpleNamespace(), "qwen_image")
        config = make_config(device="cuda", quantization=QuantizationConfig(backend="mindie"))
        with self.assertRaisesRegex(ValueError, "requires an Ascend"):
            adapter.validate_config(config)


if __name__ == "__main__":
    unittest.main()
