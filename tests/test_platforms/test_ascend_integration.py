import os
import unittest

import torch


RUN_ASCEND_TESTS = os.getenv("RUN_ASCEND_TESTS", "0") == "1"
RUN_ASCEND_QUANT_TESTS = os.getenv("RUN_ASCEND_QUANT_TESTS", "0") == "1"


@unittest.skipUnless(RUN_ASCEND_TESTS, "RUN_ASCEND_TESTS is not set")
class TestAscendOperatorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch_npu  # noqa: F401

        from diffsynth_engine.platforms import probe_ascend_feature

        cls.probe_feature = staticmethod(probe_ascend_feature)
        if not cls.probe_feature("device"):
            raise unittest.SkipTest("No available Ascend NPU")

    def test_basic_npu_execution(self):
        from diffsynth_engine.platforms import resolve_platform

        platform = resolve_platform("npu:0")
        platform.set_device("npu:0")
        lhs = torch.randn(32, 32, device="npu:0", dtype=torch.bfloat16)
        rhs = torch.randn(32, 32, device="npu:0", dtype=torch.bfloat16)
        output = lhs @ rhs
        platform.synchronize()
        self.assertEqual(output.device.type, "npu")
        self.assertEqual(output.shape, (32, 32))

    def test_mindie_attention_operator(self):
        if not self.probe_feature("mindie_attention"):
            self.skipTest("MindIE attention is unavailable")
        from diffsynth_engine.models.basic.attention import attention

        q = torch.randn(1, 128, 8, 128, device="npu:0", dtype=torch.bfloat16)
        output = attention(q, q, q, attn_impl="mindie")
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(output.device.type, "npu")

    @unittest.skipUnless(RUN_ASCEND_QUANT_TESTS, "RUN_ASCEND_QUANT_TESTS is not set")
    def test_native_linear_quantization_operators(self):
        self.assertTrue(self.probe_feature("mindie_mxfp8_linear"))
        self.assertTrue(self.probe_feature("mindie_w4a4_linear"))
        import torch.nn as nn
        from mindiesd.quantization import OnlineQuantConfig, quantize
        from mindiesd.quantization.mode import QuantAlgorithm

        for algorithm in (QuantAlgorithm.W8A8_MXFP8, QuantAlgorithm.W4A4_MXFP4_DYNAMIC):
            with self.subTest(algorithm=algorithm):
                model = nn.Sequential(nn.Linear(256, 256, device="npu:0", dtype=torch.bfloat16))
                model = quantize(
                    model,
                    online_config=OnlineQuantConfig(quant_type=algorithm),
                    dtype=torch.bfloat16,
                )
                output = model(torch.randn(1, 256, device="npu:0", dtype=torch.bfloat16))
                self.assertEqual(output.shape, (1, 256))
                self.assertTrue(torch.isfinite(output).all().cpu())

    @unittest.skipUnless(RUN_ASCEND_QUANT_TESTS, "RUN_ASCEND_QUANT_TESTS is not set")
    def test_native_fp8_attention_operator(self):
        self.assertTrue(self.probe_feature("mindie_fp8_attention"))
        import torch.nn as nn
        from mindiesd.quantization import OnlineQuantConfig, quantize
        from mindiesd.quantization.mode import QuantAlgorithm

        class ProbeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.head_dim = 128
                self.register_buffer("_device_anchor", torch.empty(0, device="npu:0"))

        model = nn.ModuleDict({"attn": ProbeAttention()}).to("npu:0")
        model = quantize(
            model,
            online_config=OnlineQuantConfig(
                quant_type=QuantAlgorithm.W8A8_DYNAMIC,
                fa_layers=("ProbeAttention",),
                fa_quant_type=QuantAlgorithm.FP8_DYNAMIC,
            ),
            dtype=torch.bfloat16,
        )
        q = torch.randn(1, 128, 8, 128, device="npu:0", dtype=torch.bfloat16)
        output = model["attn"].fa_quant(q, q, q, layout="BSND")
        self.assertEqual(output.shape, q.shape)
        self.assertTrue(torch.isfinite(output).all().cpu())


if __name__ == "__main__":
    unittest.main()
