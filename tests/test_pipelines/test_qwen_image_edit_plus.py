import glob
import os
import unittest

import torch

from diffsynth_engine import QwenImagePipelineConfig
from diffsynth_engine.pipelines import QwenImagePipeline
from diffsynth_engine.utils.download import fetch_model
from tests.common.test_case import ImageTestCase

# Multi-device runs need a local copy of the weights and an explicit parallel degree.
# Everything defaults to the upstream behaviour: one device, weights from ModelScope.
#
#   DSNP_MODEL_ROOT   local weight dir containing transformer/ text_encoder/ vae/
#   DSNP_MODEL_ID     model id used when DSNP_MODEL_ROOT is unset
#   DSNP_PARALLELISM  total number of devices, default 1
#   DSNP_SP_ULYSSES   Ulysses degree
#   DSNP_SP_RING      ring degree
#   DSNP_DEVICE       device string, defaults to cuda when available, otherwise npu
#   DSNP_FP8_LINEAR   set to 1 to quantize the DiT Linear layers to W8A8
MODEL_ROOT = os.environ.get("DSNP_MODEL_ROOT", "")
MODEL_ID = os.environ.get("DSNP_MODEL_ID", "Qwen/Qwen-Image-Edit-2509")
PARALLELISM = int(os.environ.get("DSNP_PARALLELISM", "1"))
SP_ULYSSES = int(os.environ.get("DSNP_SP_ULYSSES", "1"))
SP_RING = int(os.environ.get("DSNP_SP_RING", "1"))
FP8_LINEAR = os.environ.get("DSNP_FP8_LINEAR", "0") == "1"
DEVICE = os.environ.get("DSNP_DEVICE", "cuda" if torch.cuda.is_available() else "npu")


def _resolve_weight(sub: str):
    """Prefer a local directory, fall back to downloading from ModelScope."""
    if MODEL_ROOT:
        files = sorted(glob.glob(os.path.join(MODEL_ROOT, sub, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no safetensors under {MODEL_ROOT}/{sub}")
        return files if len(files) > 1 else files[0]
    return fetch_model(MODEL_ID, path=f"{sub}/*.safetensors")


class TestQwenImageEditPlusPipeline(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        # In parallel mode the device must not carry an index: every spawned worker calls
        # set_device(rank) first, so "npu:0" would put all of them on device 0 and OOM.
        device = DEVICE
        if PARALLELISM > 1 and ":" in device:
            raise ValueError(f"device must not carry an index in parallel mode (got {device!r})")

        config = QwenImagePipelineConfig(
            model_path=_resolve_weight("transformer"),
            encoder_path=_resolve_weight("text_encoder"),
            vae_path=_resolve_weight("vae"),
            model_dtype=torch.bfloat16,
            encoder_dtype=torch.bfloat16,
            vae_dtype=torch.float32,
            device=device,
            use_fp8_linear=FP8_LINEAR,
            parallelism=PARALLELISM,
            sp_ulysses_degree=SP_ULYSSES,
            sp_ring_degree=SP_RING,
        )
        cls.pipe = QwenImagePipeline.from_pretrained(config)

    @classmethod
    def tearDownClass(cls):
        del cls.pipe

    def test_txt2img(self):
        image = self.pipe(
            prompt="根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而庄重。",
            input_image=[self.get_input_image("qwen_1.png"), self.get_input_image("qwen_2.png")],
            negative_prompt=" ",
            cfg_scale=4.0,
            num_inference_steps=40,
            seed=42,
        )
        self.assertImageEqualAndSaveFailed(image, "qwen_image/qwen_image_edit_plus.png", threshold=0.95)


if __name__ == "__main__":
    unittest.main()
