import sys
import types
import unittest
from unittest import mock

import torch

from diffsynth_engine.models.basic import attention as attention_ops
from diffsynth_engine.models.basic import attention_backend
from diffsynth_engine.models.basic.attention_backend import (
    AttentionBackend,
    AttentionRequest,
    register_attention_backend,
    resolve_attention_backend,
)


class FakeTensor:
    def __init__(self, device_type="cpu", shape=(1, 8, 2, 4)):
        self.device = types.SimpleNamespace(type=device_type)
        self.shape = shape
        self.ndim = len(shape)


def make_request(device_type="cpu"):
    tensor = FakeTensor(device_type)
    return AttentionRequest(tensor, tensor, tensor)


class TestAttentionBackendResolution(unittest.TestCase):
    def test_npu_auto_order_is_mindie_sdpa_eager(self):
        backends = (
            AttentionBackend("mindie", mock.Mock(), frozenset({"npu"}), 800),
            AttentionBackend("sdpa", mock.Mock(), None, 300),
            AttentionBackend("eager", mock.Mock(), None, 100),
        )
        with mock.patch.dict(attention_backend._ATTENTION_BACKENDS, {}, clear=True):
            for backend in backends:
                register_attention_backend(backend)
            self.assertEqual(resolve_attention_backend("auto", make_request("npu")).name, "mindie")

            register_attention_backend(
                AttentionBackend("mindie", mock.Mock(), frozenset({"npu"}), 800, lambda: False),
                overwrite=True,
            )
            self.assertEqual(resolve_attention_backend("auto", make_request("npu")).name, "sdpa")

            register_attention_backend(
                AttentionBackend("sdpa", mock.Mock(), None, 300, lambda: False),
                overwrite=True,
            )
            self.assertEqual(resolve_attention_backend("auto", make_request("npu")).name, "eager")

    def test_explicit_unavailable_backend_does_not_fall_back(self):
        backend = AttentionBackend("optional", mock.Mock(), None, 1, lambda: False)
        with mock.patch.dict(attention_backend._ATTENTION_BACKENDS, {"optional": backend}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "not available"):
                resolve_attention_backend("optional", make_request())

    def test_auto_can_preserve_backend_specific_legacy_capabilities(self):
        explicit_support = mock.Mock(return_value=(False, "explicit request rejected"))
        auto_support = mock.Mock(return_value=(True, None))
        backend = AttentionBackend(
            "legacy",
            mock.Mock(),
            supports=explicit_support,
            auto_supports=auto_support,
        )
        with mock.patch.dict(attention_backend._ATTENTION_BACKENDS, {"legacy": backend}, clear=True):
            self.assertIs(resolve_attention_backend("auto", make_request()), backend)
            with self.assertRaisesRegex(RuntimeError, "explicit request rejected"):
                resolve_attention_backend("legacy", make_request())

    def test_cuda_auto_priority_matches_mainline(self):
        expected = ["fa4", "fa3", "aiter", "xformers", "sdpa", "fa2", "eager"]
        priorities = {
            name: attention_backend._ATTENTION_BACKENDS[name].priority
            for name in expected
        }
        self.assertEqual(
            [name for name, _ in sorted(priorities.items(), key=lambda item: item[1], reverse=True)],
            expected,
        )


class TestMindieAttention(unittest.TestCase):
    def test_forwards_public_attention_arguments(self):
        attention_forward = mock.Mock(return_value=torch.ones(1))
        module_name = "mindiesd.layers.flash_attn.attention_forward"
        fake_module = types.ModuleType(module_name)
        fake_module.attention_forward = attention_forward
        q = torch.randn(1, 4, 2, 8)
        k = torch.randn(1, 4, 2, 8)
        v = torch.randn(1, 4, 2, 8)
        mask = torch.randn(1, 1, 4, 4)
        request = AttentionRequest(q, k, v, attn_mask=mask, scale=0.125)

        with mock.patch.dict(sys.modules, {module_name: fake_module}):
            output = attention_ops._run_mindie(request)

        self.assertEqual(output.shape, (1,))
        attention_forward.assert_called_once_with(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
            scale=0.125,
            fused=True,
            head_first=False,
        )

    def test_mindie_rejects_non_4d_inputs(self):
        request = make_request("npu")
        request.q.ndim = 3
        supported, reason = attention_ops._mindie_supports(request)
        self.assertFalse(supported)
        self.assertIn("4D", reason)

    def test_explicit_mindie_has_actionable_dependency_error(self):
        with mock.patch("diffsynth_engine.platforms.AscendPlatform.supports", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "install a compatible MindIE-SD 3.x"):
                resolve_attention_backend("mindie", make_request("npu"))


if __name__ == "__main__":
    unittest.main()
