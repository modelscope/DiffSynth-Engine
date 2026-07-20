import unittest
from unittest import mock

import torch
import torch.nn as nn

from diffsynth_engine.models.qwen_image.qwen_image_dit import QwenDoubleStreamAttention
from diffsynth_engine.platforms.ascend_qwen import MindIEFP8AttentionProcessor


class FakeFAQuant(nn.Module):
    def __init__(self):
        super().__init__()
        self.call = mock.Mock()

    def forward(self, query, key, value, **kwargs):
        self.call(query, key, value, **kwargs)
        return query


class TestQwenAttentionProcessor(unittest.TestCase):
    def setUp(self):
        self.attention = QwenDoubleStreamAttention(
            dim_a=8,
            dim_b=8,
            num_heads=2,
            head_dim=4,
            device="cpu",
            dtype=torch.float32,
        )
        self.image = torch.randn(1, 3, 8)
        self.text = torch.randn(1, 2, 8)

    def test_fa_quant_processor_uses_bsnd_layout(self):
        fa_quant = FakeFAQuant()
        self.attention.set_attention_processor(MindIEFP8AttentionProcessor(fa_quant))
        image, text = self.attention(self.image, self.text)

        self.assertEqual(image.shape, self.image.shape)
        self.assertEqual(text.shape, self.text.shape)
        self.assertEqual(fa_quant.call.call_args.kwargs, {"layout": "BSND"})
        query = fa_quant.call.call_args.args[0]
        self.assertEqual(query.shape, (1, 5, 2, 4))

    def test_default_path_still_uses_common_attention(self):
        with mock.patch(
            "diffsynth_engine.models.qwen_image.qwen_image_dit.attention_ops.attention",
            side_effect=lambda q, k, v, **kwargs: q,
        ) as common_attention:
            self.attention(self.image, self.text, attn_kwargs={"attn_impl": "eager"})

        self.assertEqual(common_attention.call_count, 1)
        self.assertEqual(common_attention.call_args.kwargs["attn_impl"], "eager")

    def test_fa_quant_processor_rejects_entity_mask(self):
        self.attention.set_attention_processor(MindIEFP8AttentionProcessor(FakeFAQuant()))
        mask = torch.zeros(1, 1, 5, 5)
        with self.assertRaisesRegex(RuntimeError, "entity attention masks"):
            self.attention(self.image, self.text, attn_mask=mask)


if __name__ == "__main__":
    unittest.main()
