import unittest
import os
import json
import torch

from diffsynth_engine.models.qwen_image import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)
from diffsynth_engine.tokenizers import Qwen2TokenizerFast
from diffsynth_engine.utils.constants import (
    QWEN_IMAGE_TOKENIZER_CONF_PATH,
    QWEN_IMAGE_CONFIG_FILE,
    QWEN_IMAGE_VISION_CONFIG_FILE,
)
from diffsynth_engine.utils.download import ensure_directory_exists, fetch_model
from diffsynth_engine.utils.loader import save_file
from tests.common.test_case import TestCase, RUN_EXTRA_TEST
from tests.common.utils import load_model_checkpoint


class TestQwen2_5_VL(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Qwen2TokenizerFast.from_pretrained(QWEN_IMAGE_TOKENIZER_CONF_PATH)

        cls._model_path = fetch_model("Qwen/Qwen2.5-VL-7B-Instruct", fetch_safetensors=False)
        ckpt_path = [
            os.path.join(cls._model_path, file) for file in os.listdir(cls._model_path) if file.endswith(".safetensors")
        ]
        loaded_state_dict = load_model_checkpoint(ckpt_path, device="cpu", dtype=torch.bfloat16)
        with open(QWEN_IMAGE_VISION_CONFIG_FILE, "r") as f:
            vision_config = Qwen2_5_VLVisionConfig(**json.load(f))
        with open(QWEN_IMAGE_CONFIG_FILE, "r") as f:
            text_config = Qwen2_5_VLConfig(**json.load(f))
        cls.encoder = Qwen2_5_VLForConditionalGeneration.from_state_dict(
            loaded_state_dict,
            vision_config=vision_config,
            config=text_config,
            device="cuda:0",
            dtype=torch.bfloat16,
        ).eval()
        cls.texts = ["Hello, World!", "DiffSynth-Engine developed by Muse AI+Modelscope"]

    def test_encoder(self):
        outputs = self.tokenizer(self.texts)
        text_ids, attention_mask = outputs["input_ids"].to("cuda:0"), outputs["attention_mask"].to("cuda:0")
        with torch.no_grad():
            logits = self.encoder(input_ids=text_ids, attention_mask=attention_mask)["logits"].cpu()
        expected_tensors = self.get_expect_tensor("qwen_image/qwen2_5_vl.safetensors")
        self.assertTensorEqual(logits, expected_tensors["logits"])

    @unittest.skipUnless(RUN_EXTRA_TEST, "RUN_EXTRA_TEST is not set")
    def test_and_save_tensors(self):
        from transformers import Qwen2_5_VLForConditionalGeneration

        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._model_path, device_map="cuda:0", torch_dtype=torch.bfloat16
        ).eval()
        outputs = self.tokenizer(self.texts)
        text_ids, attention_mask = outputs["input_ids"].to("cuda:0"), outputs["attention_mask"].to("cuda:0")
        with torch.no_grad():
            expected = vlm(input_ids=text_ids, attention_mask=attention_mask).logits.cpu()
            logits = self.encoder(input_ids=text_ids, attention_mask=attention_mask)["logits"].cpu()
        self.assertTensorEqual(logits, expected)

        excepted_tensors = {"logits": logits}
        save_path = self.testdata_dir / "expect/qwen_image/qwen2_5_vl.safetensors"
        ensure_directory_exists(save_path)
        save_file(excepted_tensors, save_path)


if __name__ == "__main__":
    unittest.main()
