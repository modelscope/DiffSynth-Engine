from enum import Enum

import torch
from diffusers.loaders import lora_conversion_utils as lcu

from diffsynth_engine.layers.lora import LoRAWeights
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


class PrefixFormat(Enum):
    STANDARD = "standard"
    WAN = "wan"
    NON_DIFFUSERS_SD = "non_diffusers_sd"


class SuffixFormat(Enum):
    STANDARD = "standard"  # .lora_A.weight / .lora_B.weight
    PEFT = "peft"  # .lora_A.default.weight / .lora_B.default.weight
    QWEN_IMAGE = "qwen_image"  # .lora.down.weight / .lora.up.weight
    NON_DIFFUSERS_SD = "non_diffusers_sd"  # .lora_down.weight / .lora_up.weight


def _strip_prefix(key: str, *prefixes: str) -> str:
    for p in prefixes:
        if key.startswith(p):
            return key[len(p) :]
    return key


def _try_diffusers_maybe_convert(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
    try:
        if hasattr(lcu, "maybe_convert_state_dict"):
            converted = lcu.maybe_convert_state_dict(state_dict)
        else:
            return None
        if not isinstance(converted, dict):
            converted = dict(converted)
        return converted
    except Exception as e:
        logger.warning(f"diffusers maybe_convert_state_dict failed: {e}")
        return None


# ==============================================================================
# Detection
# ==============================================================================


def detect_prefix_format(state_dict: dict[str, torch.Tensor]) -> PrefixFormat:
    """Detect the prefix format used by a LoRA state dict.

    Args:
        state_dict: Raw LoRA state dict loaded from a weights file.

    Returns:
        Detected prefix format.
    """
    keys = state_dict.keys()

    if any(
        k.startswith("diffusion_model.blocks.")
        and (".cross_attn." in k or ".self_attn." in k or ".ffn." in k or ".norm3." in k)
        for k in keys
    ):
        return PrefixFormat.WAN

    if all(k.startswith(("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_")) for k in keys):
        return PrefixFormat.NON_DIFFUSERS_SD

    return PrefixFormat.STANDARD


def detect_suffix_format(state_dict: dict[str, torch.Tensor]) -> SuffixFormat:
    """Detect the suffix format used by LoRA down/up weight keys.

    Args:
        state_dict: Raw LoRA state dict loaded from a weights file.

    Returns:
        Detected suffix format.

    Raises:
        ValueError: If none of the supported LoRA suffix patterns are found.
    """
    for k in state_dict.keys():
        if k.endswith(".lora_A.default.weight") or k.endswith(".lora_B.default.weight"):
            return SuffixFormat.PEFT
        if k.endswith(".lora.down.weight") or k.endswith(".lora.up.weight"):
            return SuffixFormat.QWEN_IMAGE
        if k.endswith(".lora_down.weight") or k.endswith(".lora_up.weight"):
            return SuffixFormat.NON_DIFFUSERS_SD
        if k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"):
            return SuffixFormat.STANDARD

    raise ValueError(
        "Could not detect LoRA suffix format. "
        "Supported suffixes: "
        ".lora_A.weight/.lora_B.weight (standard), "
        ".lora_A.default.weight/.lora_B.default.weight (peft), "
        ".lora.down.weight/.lora.up.weight (qwen_image), "
        ".lora_down.weight/.lora_up.weight (non_diffusers_sd)"
    )


# ==============================================================================
# Conversion
# ==============================================================================


def _convert_suffix(state_dict: dict[str, torch.Tensor], suffix_fmt: SuffixFormat) -> dict[str, torch.Tensor]:
    """Normalize suffixes to .lora_A.weight / .lora_B.weight."""

    if suffix_fmt == SuffixFormat.PEFT:
        return {
            k.replace("lora_A.default", "lora_A").replace("lora_B.default", "lora_B"): v for k, v in state_dict.items()
        }

    if suffix_fmt == SuffixFormat.QWEN_IMAGE:
        return {
            k.replace(".lora.down.weight", ".lora_A.weight").replace(".lora.up.weight", ".lora_B.weight"): v
            for k, v in state_dict.items()
        }

    if suffix_fmt == SuffixFormat.NON_DIFFUSERS_SD:
        return {
            k.replace(".lora_down.weight", ".lora_A.weight").replace(".lora_up.weight", ".lora_B.weight"): v
            for k, v in state_dict.items()
        }

    return state_dict


def _convert_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix_fmt: PrefixFormat,
) -> dict[str, torch.Tensor]:
    """Apply prefix/body conversion for the detected prefix format."""
    converted = _try_diffusers_maybe_convert(state_dict)
    if converted is None:
        converted = _convert_wan(state_dict) if prefix_fmt == PrefixFormat.WAN else state_dict
    return _convert_standard(converted)


def _convert_wan(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    body_map = {
        ".self_attn.q.": ".attn1.to_q.",
        ".self_attn.k.": ".attn1.to_k.",
        ".self_attn.v.": ".attn1.to_v.",
        ".self_attn.o.": ".attn1.to_out.0.",
        ".cross_attn.q.": ".attn2.to_q.",
        ".cross_attn.k.": ".attn2.to_k.",
        ".cross_attn.v.": ".attn2.to_v.",
        ".cross_attn.o.": ".attn2.to_out.0.",
        ".ffn.0.": ".ffn.net.0.proj.",
        ".ffn.2.": ".ffn.net.2.",
    }
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        k = _strip_prefix(k, "diffusion_model.")
        for old, new in body_map.items():
            k = k.replace(old, new)
        out[k] = v
    return out


def _convert_standard(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        out[
            _strip_prefix(k, "base_model.model.transformer.", "base_model.model.", "transformer.", "diffusion_model.")
        ] = v
    return out


def convert_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, LoRAWeights]:
    """Convert a raw LoRA state dict into per-layer ``LoRAWeights``.

    The converter detects the input key prefix and suffix formats, normalizes
    keys to the internal ``.lora_A.weight`` / ``.lora_B.weight`` form, and builds
    one ``LoRAWeights`` object for each target layer.

    Args:
        state_dict: Raw LoRA state dict loaded from a weights file.

    Returns:
        Mapping from target layer name to ``LoRAWeights``.

    Raises:
        ValueError: If a down-projection key is found without the matching
            up-projection key.
    """
    prefix_fmt = detect_prefix_format(state_dict)
    suffix_fmt = detect_suffix_format(state_dict)
    logger.info(f"Detected LoRA formats: prefix={prefix_fmt.value}, suffix={suffix_fmt.value}.")

    state_dict = _convert_prefix(state_dict, prefix_fmt)
    state_dict = _convert_suffix(state_dict, suffix_fmt)

    lora_weights_dict: dict[str, LoRAWeights] = {}  # layer_name -> LoRAWeights
    for key in list(state_dict.keys()):
        if not key.endswith(".lora_A.weight"):
            continue

        base_key = key[: -len(".lora_A.weight")]
        up_key = base_key + ".lora_B.weight"
        if up_key not in state_dict:
            raise ValueError(f"Found down key '{key}' but missing up key '{up_key}'")

        alpha_key = base_key + ".alpha"
        down = state_dict[key]
        up = state_dict[up_key]
        rank = up.shape[1]
        alpha = state_dict[alpha_key] if alpha_key in state_dict else rank
        # Ensure alpha is a scalar (int or float), not a tensor
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.item() if alpha.numel() == 1 else alpha

        lora_weights_dict[base_key] = LoRAWeights(
            down=down,
            up=up,
            rank=rank,
            alpha=float(alpha),
        )

    return lora_weights_dict
