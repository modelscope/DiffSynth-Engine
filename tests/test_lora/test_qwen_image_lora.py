"""Unit tests for LoRA operations via DiffSynthEngine with QwenImagePipeline."""

import os
import unittest

import torch

from diffsynth_engine.configs import QwenImagePipelineConfig
from diffsynth_engine.engine import DiffSynthEngine
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase


class TestQwenImageLoRA(ImageTestCase):
    """Sequential LoRA API tests. Each API parameter form is exercised exactly once."""

    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Qwen/Qwen-Image")
        cls.lora_a_dir = fetch_model("DiffSynth-Studio/Qwen-Image-Distill-LoRA")
        cls.lora_b_dir = fetch_model("DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1")

        config = QwenImagePipelineConfig(
            model_path=model_path,
            device="cuda",
            model_dtype=torch.bfloat16,
        )
        cls.engine = DiffSynthEngine.from_pretrained(config)

        cls.prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
        cls.width = 512
        cls.height = 512
        cls.steps = 15
        cls.seed = 0

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def _loras(self):
        return [
            {
                "lora_id": "lora_a",
                "path": os.path.join(self.lora_a_dir, "model.safetensors"),
                "scale": 0.8,
            },
            {
                "lora_id": "lora_b",
                "path": os.path.join(self.lora_b_dir, "model.safetensors"),
                "scale": 0.8,
            },
        ]

    def _generate(self):
        return self.engine.generate(
            prompt=self.prompt,
            width=self.width,
            height=self.height,
            true_cfg_scale=1.0,
            num_inference_steps=self.steps,
            generator=torch.Generator(device="cuda").manual_seed(self.seed),
        )

    def test_01_load_and_list(self):
        """load_loras(dict), load_loras(list); list_loras(None/str/list)."""
        self.engine.reset_loras()

        # single dict
        [id_a] = self.engine.load_loras(self._loras()[0])
        self.assertEqual(id_a, "lora_a")

        # list of dicts
        [id_b] = self.engine.load_loras([self._loras()[1]])
        self.assertEqual(id_b, "lora_b")

        # list_loras variants
        self.assertEqual(len(self.engine.list_loras()), 2)
        self.assertEqual(self.engine.list_loras("lora_a")[0]["lora_id"], "lora_a")
        self.assertEqual(len(self.engine.list_loras(["lora_a", "lora_b"])), 2)

    def test_02_deactivate_and_activate(self):
        """deactivate_loras(str/list/None); activate_loras(str+float, list+list)."""
        # state: both active from test_01
        self.engine.deactivate_loras("lora_a")
        self.assertEqual(self.engine.list_loras("lora_a")[0]["status"], "inactive")
        self.assertEqual(self.engine.list_loras("lora_b")[0]["status"], "active")

        self.engine.deactivate_loras(["lora_b"])
        self.assertEqual(self.engine.list_loras("lora_b")[0]["status"], "inactive")

        # activate: str + float scale
        self.engine.activate_loras("lora_a", scales=0.5)
        lora_a = self.engine.list_loras("lora_a")[0]
        self.assertEqual(lora_a["status"], "active")
        self.assertEqual(lora_a["scale"], 0.5)

        # activate: list + list of scales
        self.engine.activate_loras(["lora_b"], scales=[0.6])
        lora_b = self.engine.list_loras("lora_b")[0]
        self.assertEqual(lora_b["status"], "active")
        self.assertEqual(lora_b["scale"], 0.6)

        # deactivate None (all)
        self.engine.deactivate_loras()
        loras = self.engine.list_loras()
        self.assertTrue(all(item["status"] == "inactive" for item in loras))

        image = self._generate().images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image_lora/qwen_image_lora_base.png")

    def test_03_set_active(self):
        """set_active_loras(str/None, str/float, list/list, list/float)."""
        # state: both inactive from test_02
        self.engine.set_active_loras("lora_a")
        self.assertEqual(self.engine.list_loras("lora_a")[0]["status"], "active")
        self.assertEqual(self.engine.list_loras("lora_b")[0]["status"], "inactive")

        self.engine.set_active_loras("lora_b", scales=0.5)
        self.assertEqual(self.engine.list_loras("lora_a")[0]["status"], "inactive")
        self.assertEqual(self.engine.list_loras("lora_b")[0]["scale"], 0.5)

        self.engine.set_active_loras(["lora_a", "lora_b"], scales=[0.3, 0.9])
        loras = {item["lora_id"]: item for item in self.engine.list_loras()}
        self.assertEqual(loras["lora_a"]["scale"], 0.3)
        self.assertEqual(loras["lora_b"]["scale"], 0.9)

        self.engine.set_active_loras(["lora_a", "lora_b"], scales=0.8)
        loras = {item["lora_id"]: item for item in self.engine.list_loras()}
        self.assertEqual(loras["lora_a"]["scale"], 0.8)
        self.assertEqual(loras["lora_b"]["scale"], 0.8)

        image = self._generate().images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image_lora/qwen_image_lora_stacked_ab.png")

    def test_04_merge_and_unmerge(self):
        """merge_loras() default; unmerge_loras() default."""
        # state: both active at 0.8 from test_03
        self.engine.merge_loras()
        loras = self.engine.list_loras()
        self.assertTrue(all(item["status"] == "merged" for item in loras))

        image = self._generate().images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image_lora/qwen_image_lora_stacked_ab.png")

        self.engine.unmerge_loras()
        self.assertEqual(self.engine.list_loras(), [])

    def test_05_merge_with_params(self):
        """merge_loras(target_module, chunked, high_precision); unmerge_loras(target_module)."""
        # state: empty after test_04 unmerge
        self.engine.load_loras(self._loras()[0])
        self.engine.merge_loras(target_module="transformer", chunked=True, high_precision=False)
        self.assertEqual(self.engine.list_loras("lora_a")[0]["status"], "merged")

        image = self._generate().images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image_lora/qwen_image_lora_single_a.png", threshold=0.90)

        self.engine.unmerge_loras(target_module="transformer")
        self.assertEqual(self.engine.list_loras(), [])

    def test_06_unload(self):
        """unload_loras(str); unload_loras(None)."""
        # state: empty after test_05
        [id_a, id_b] = self.engine.load_loras(self._loras())

        self.engine.unload_loras(id_a)
        loras = self.engine.list_loras()
        self.assertEqual(len(loras), 1)
        self.assertEqual(loras[0]["lora_id"], id_b)

        self.engine.unload_loras(None)
        self.assertEqual(self.engine.list_loras(), [])

    def test_07_reset(self):
        """reset_loras(target_module); reset_loras(None)."""
        # state: empty after test_06
        self.engine.load_loras(self._loras())
        self.engine.merge_loras()
        self.engine.reset_loras(target_module="transformer")
        self.assertEqual(self.engine.list_loras(), [])

        self.engine.load_loras(self._loras())
        self.engine.reset_loras()
        self.assertEqual(self.engine.list_loras(), [])

        image = self._generate().images[0]
        self.assertImageEqualAndSaveFailed(image, "qwen_image_lora/qwen_image_lora_base.png")


class TestQwenImageLoRAMultiWorker(ImageTestCase):
    """Basic LoRA lifecycle test with multi-worker (parallelism=2)."""

    @classmethod
    def setUpClass(cls):
        model_path = fetch_model("Qwen/Qwen-Image")
        cls.lora_a_dir = fetch_model("DiffSynth-Studio/Qwen-Image-Distill-LoRA")

        config = QwenImagePipelineConfig(
            model_path=model_path,
            device="cuda",
            model_dtype=torch.bfloat16,
            parallelism=2,
        )
        cls.engine = DiffSynthEngine.from_pretrained(config)

        cls.prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
        cls.width = 512
        cls.height = 512
        cls.steps = 15
        cls.seed = 0

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def _generate(self):
        return self.engine.generate(
            prompt=self.prompt,
            width=self.width,
            height=self.height,
            true_cfg_scale=1.0,
            num_inference_steps=self.steps,
            generator=torch.Generator(device="cuda").manual_seed(self.seed),
        )

    def test_01_full_lifecycle(self):
        """load → activate → set_active → merge → unmerge → unload → reset."""
        self.engine.reset_loras()
        lora = {
            "lora_id": "lora_a",
            "path": os.path.join(self.lora_a_dir, "model.safetensors"),
            "scale": 0.8,
        }
        [lora_id] = self.engine.load_loras(lora)
        self.assertEqual(self.engine.list_loras()[0]["status"], "active")

        self.engine.deactivate_loras()
        self.assertEqual(self.engine.list_loras(lora_id)[0]["status"], "inactive")

        self.engine.activate_loras(lora_id, scales=0.5)
        self.assertEqual(self.engine.list_loras(lora_id)[0]["scale"], 0.5)

        self.engine.set_active_loras(lora_id, scales=0.8)
        self.assertEqual(self.engine.list_loras(lora_id)[0]["status"], "active")

        self.engine.merge_loras()
        self.assertEqual(self.engine.list_loras(lora_id)[0]["status"], "merged")

        self.engine.unmerge_loras()
        self.assertEqual(self.engine.list_loras(lora_id), [])

        [lora_id] = self.engine.load_loras(lora)
        self.engine.unload_loras(lora_id)
        self.assertEqual(self.engine.list_loras(), [])

        self.engine.load_loras(lora)
        self.engine.reset_loras()
        self.assertEqual(self.engine.list_loras(), [])

        image = self._generate().images[0]
        self.assertEqual(image.size, (self.width, self.height))


if __name__ == "__main__":
    unittest.main()
