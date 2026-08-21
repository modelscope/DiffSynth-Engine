"""NPU 单卡 Qwen Image 全场景集成测试"""
import os
import unittest

import torch

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import QwenImagePipelineConfig
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase

# NPU 可用性检查
try:
    import torch_npu  # noqa: F401

    NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    NPU_AVAILABLE = False


@unittest.skipUnless(NPU_AVAILABLE, "NPU not available")
class TestQwenImageNPU(ImageTestCase):
    """NPU 单卡 text-to-image 测试"""

    @classmethod
    def setUpClass(cls):
        os.environ["USE_MINDIESD_FUSE"] = "true"
        cls.model_path = fetch_model("Qwen/Qwen-Image")
        config = QwenImagePipelineConfig(
            model_path=cls.model_path,
            device="npu",
            attn_type="mindie",
            model_dtype=torch.bfloat16,
        )
        cls.engine = DiffSynthEngine.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        del cls.engine
        torch.npu.empty_cache()

    def test_txt2img(self):
        prompt = "A painting of a cat in a zen garden"
        negative_prompt = "ugly, blurry, low quality"
        output = self.engine.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=4.0,
            width=1328,
            height=1328,
            num_inference_steps=28,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        image = output.images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image.png", threshold=0.95)


@unittest.skipUnless(NPU_AVAILABLE, "NPU not available")
class TestQwenImageEditNPU(ImageTestCase):
    """NPU 单卡 image-edit 测试"""

    @classmethod
    def setUpClass(cls):
        os.environ["USE_MINDIESD_FUSE"] = "true"
        cls.model_path = fetch_model("Qwen/Qwen-Image-Edit")
        config = QwenImagePipelineConfig(
            model_path=cls.model_path,
            device="npu",
            attn_type="mindie",
            model_dtype=torch.bfloat16,
        )
        cls.engine = DiffSynthEngine.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        del cls.engine
        torch.npu.empty_cache()

    def test_single_image_edit(self):
        """Test single image editing on NPU"""
        input_image = self.get_input_image("qwen_image_edit_input.png")
        prompt = "Replace '通义千问' with '呜哩AI'"
        negative_prompt = " "

        output = self.engine.generate(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=4.0,
            num_inference_steps=50,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        image = output.images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image_edit.png", threshold=0.95)


@unittest.skipUnless(NPU_AVAILABLE, "NPU not available")
class TestQwenImageEditPlusNPU(ImageTestCase):
    """NPU 单卡 edit-plus 测试"""

    @classmethod
    def setUpClass(cls):
        os.environ["USE_MINDIESD_FUSE"] = "true"
        cls.model_path = fetch_model("Qwen/Qwen-Image-Edit-2511")
        config = QwenImagePipelineConfig(
            model_path=cls.model_path,
            device="npu",
            attn_type="mindie",
            model_dtype=torch.bfloat16,
        )
        cls.engine = DiffSynthEngine.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        del cls.engine
        torch.npu.empty_cache()

    def test_single_image_edit(self):
        """Test single image editing with Edit Plus pipeline on NPU"""
        input_image = self.get_input_image("qwen_image_edit_input.png")
        prompt = "Replace '通义千问' with '呜哩AI'"
        negative_prompt = " "

        output = self.engine.generate(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=4.0,
            num_inference_steps=50,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        image = output.images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image_edit_plus_single_2511.png", threshold=0.95)

    def test_multi_image_edit(self):
        """Test multiple images editing with Edit Plus pipeline on NPU"""
        input_images = [
            self.get_input_image("qwen_image_edit_input_1.png").convert("RGB"),
            self.get_input_image("qwen_image_edit_input_2.png").convert("RGB"),
        ]
        prompt = "根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而庄重。"
        negative_prompt = " "

        output = self.engine.generate(
            image=input_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=4.0,
            num_inference_steps=40,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        image = output.images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image_edit_plus_multi_2511.png", threshold=0.95)


@unittest.skipUnless(NPU_AVAILABLE, "NPU not available")
class TestQwenImageLayeredNPU(ImageTestCase):
    """NPU 单卡 layered 测试"""

    @classmethod
    def setUpClass(cls):
        os.environ["USE_MINDIESD_FUSE"] = "true"
        cls.model_path = fetch_model("Qwen/Qwen-Image-Layered")
        config = QwenImagePipelineConfig(
            model_path=cls.model_path,
            device="npu",
            attn_type="mindie",
            model_dtype=torch.bfloat16,
        )
        cls.engine = DiffSynthEngine.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        del cls.engine
        torch.npu.empty_cache()

    def test_image_layered(self):
        """Test layered image generation on NPU"""
        input_image = self.get_input_image("qwen_image_layered_input.png").convert("RGBA")
        prompt = ""

        output = self.engine.generate(
            image=input_image,
            prompt=prompt,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            layers=3,
            resolution=640,
            cfg_normalize=False,
            use_en_prompt=True,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )

        images = output.images[0]
        self.assertEqual(len(images), 3)

        for i, layer_image in enumerate(images):
            self.assertImageEqualAndSaveFailed(
                layer_image,
                f"qwen_image/qwen_image_layered_{i}.png",
                threshold=0.95,
            )


if __name__ == "__main__":
    unittest.main()
