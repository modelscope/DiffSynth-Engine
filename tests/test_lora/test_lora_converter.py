import os
import unittest

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file

from diffsynth_engine.pipelines.lora.converter import (
    PrefixFormat,
    SuffixFormat,
    convert_lora_state_dict,
    detect_prefix_format,
    detect_suffix_format,
)
from diffsynth_engine.utils.download import fetch_model


def _download_lora(repo_id: str, filename: str) -> str:
    try:
        local_dir = fetch_model(repo_id, path=filename, source="modelscope")
    except Exception:
        local_dir = fetch_model(repo_id, path=filename, source="huggingface")
    return os.path.join(local_dir, filename)


def _get_model_linear_paths(model) -> set[str]:
    return set(name for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear))


def _build_wan_model() -> set[str]:
    from diffusers import WanTransformer3DModel

    with init_empty_weights():
        model = WanTransformer3DModel(
            num_layers=40,
            num_attention_heads=40,
            attention_head_dim=128,
        )
    return _get_model_linear_paths(model)


def _build_qwen_image_model() -> set[str]:
    from diffusers import QwenImageTransformer2DModel

    with init_empty_weights():
        model = QwenImageTransformer2DModel(
            num_layers=60,
            num_attention_heads=24,
            attention_head_dim=128,
            in_channels=64,
            out_channels=16,
            joint_attention_dim=3584,
        )
    return _get_model_linear_paths(model)


# =========================================================================
# Test cases
# =========================================================================

_TEST_CASES = {
    "wan_distill": {
        "name": "Wan2.2 Distill LoRA (lightx2v)",
        "repo_id": "lightx2v/Wan2.2-Distill-Loras",
        "filename": "wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors",
        "expected_prefix": PrefixFormat.WAN,
        "expected_suffix": SuffixFormat.NON_DIFFUSERS_SD,
        "model_builder": "_build_wan_model",
    },
    "wan_standard": {
        "name": "Wan2.2 Standard LoRA (Cseti)",
        "repo_id": "Cseti/wan2.2-14B-Arcane_Jinx-lora-v1",
        "filename": "985347-wan22_14B-low-Nfj1nx-e65.safetensors",
        "expected_prefix": PrefixFormat.WAN,
        "expected_suffix": SuffixFormat.STANDARD,
        "model_builder": "_build_wan_model",
    },
    "qwen_image_peft": {
        "name": "Qwen-Image PEFT LoRA (flymy-ai)",
        "repo_id": "flymy-ai/qwen-image-realism-lora",
        "filename": "flymy_realism.safetensors",
        "expected_prefix": PrefixFormat.STANDARD,
        "expected_suffix": SuffixFormat.QWEN_IMAGE,
        "model_builder": "_build_qwen_image_model",
    },
    "qwen_image_diffusers": {
        "name": "Qwen-Image Diffusers LoRA (ostris)",
        "repo_id": "ostris/qwen_image_edit_inpainting",
        "filename": "qwen_image_edit_inpainting.safetensors",
        "expected_prefix": PrefixFormat.STANDARD,
        "expected_suffix": SuffixFormat.STANDARD,
        "model_builder": "_build_qwen_image_model",
    },
    "qwen_image_lightning": {
        "name": "Qwen-Image Lightning LoRA (lightx2v)",
        "repo_id": "lightx2v/Qwen-Image-Lightning",
        "filename": "Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors",
        "expected_prefix": PrefixFormat.STANDARD,
        "expected_suffix": SuffixFormat.NON_DIFFUSERS_SD,
        "model_builder": "_build_qwen_image_model",
    },
}

_MODEL_CACHE: dict[str, set[str]] = {}


def _get_model_paths(builder_name: str) -> set[str]:
    if builder_name not in _MODEL_CACHE:
        _MODEL_CACHE[builder_name] = globals()[builder_name]()
    return _MODEL_CACHE[builder_name]


class TestLoRAConverterMatchesModel(unittest.TestCase):
    """Verify converter output keys exist in diffusers model named_modules()."""

    def _run_case(self, case_key: str):
        case = _TEST_CASES[case_key]
        name = case["name"]

        local_path = _download_lora(case["repo_id"], case["filename"])
        raw_state = load_file(local_path)

        # Verify prefix format detection
        prefix_fmt = detect_prefix_format(raw_state)
        self.assertEqual(
            prefix_fmt,
            case["expected_prefix"],
            f"[{name}] prefix format detection mismatch",
        )

        # Verify suffix format detection
        suffix_fmt = detect_suffix_format(raw_state)
        self.assertEqual(
            suffix_fmt,
            case["expected_suffix"],
            f"[{name}] suffix format detection mismatch",
        )

        # Convert
        lora_weights = convert_lora_state_dict(raw_state)
        self.assertGreater(len(lora_weights), 0, f"[{name}] conversion produced empty result")

        # Check against model
        model_paths = _get_model_paths(case["model_builder"])
        lora_keys = set(lora_weights.keys())
        allowed_missing = case.get("allowed_missing", set())
        missing = lora_keys - model_paths - allowed_missing

        if missing:
            sample = sorted(missing)[:10]
            available_sample = sorted(model_paths)[:10]
            self.fail(
                f"[{name}] {len(missing)}/{len(lora_keys)} converter keys "
                f"NOT found in diffusers model named_modules():\n"
                f"  Missing keys (sample): {sample}\n"
                f"  Model paths (sample): {available_sample}"
            )

    def test_wan_distill(self):
        self._run_case("wan_distill")

    def test_wan_standard(self):
        self._run_case("wan_standard")

    def test_qwen_image_peft(self):
        self._run_case("qwen_image_peft")

    def test_qwen_image_diffusers(self):
        self._run_case("qwen_image_diffusers")

    def test_qwen_image_lightning(self):
        self._run_case("qwen_image_lightning")


if __name__ == "__main__":
    unittest.main()
