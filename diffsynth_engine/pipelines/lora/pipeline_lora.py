from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch.nn as nn

from diffsynth_engine.layers.lora import LoRALayer, wrap_with_lora_layer
from diffsynth_engine.pipelines.lora.converter import convert_lora_state_dict
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.load_utils import load_safetensors

logger = logging.get_logger(__name__)


class LoRAStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MERGED = "merged"


@dataclass
class LoRARef:
    """Reference record for a LoRA model.

    Attributes:
        lora_id: Unique id for this LoRA model.
        path: Safetensors file path used to load LoRA weights.
        target_module: Pipeline module name that owns target LoRA layers.
            If None, the pipeline default target module is used.
        scale: Current scale applied to the model.
        status: Current status of the model.
    """

    lora_id: str
    path: str
    target_module: str | None = None
    scale: float = 1.0
    status: LoRAStatus | None = None


class LoRAPipeline:
    """Mixin providing LoRA lifecycle management for pipelines."""

    _lora_target_module = "transformer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_lora_modules()

    def init_lora_modules(self):
        """Initialize empty LoRA state for the pipeline."""
        self.lora_refs: dict[str, LoRARef] = {}
        self.lora_layers: dict[str, dict[str, LoRALayer]] = {}  # module_name -> layer_name -> LoRALayer

    def convert_to_lora_layers(self, module_name: str):
        """Convert all supported layers in a target module to LoRA-capable layers.

        Args:
            module_name: Name of the pipeline module to convert, for example "transformer".
        """
        if module_name in self.lora_layers:
            return

        if not hasattr(self, module_name):
            raise KeyError(f"target module '{module_name}' does not exist.")
        module = getattr(self, module_name)
        if not isinstance(module, nn.Module):
            raise TypeError(f"target module '{module_name}' must be an nn.Module, got {type(module)}.")

        self.lora_layers[module_name] = {}
        layers = self.lora_layers[module_name]

        for layer_name, layer in list(module.named_modules()):
            if not layer_name or not isinstance(layer, nn.Linear):
                continue
            lora_layer = wrap_with_lora_layer(layer)
            if lora_layer is None:
                continue
            module.set_submodule(layer_name, lora_layer, strict=True)
            layers[layer_name] = lora_layer

        logger.info(f"Converted {len(layers)} layers to LoRA layers.")

    def load_loras(
        self,
        lora_args: dict[str, Any] | list[dict[str, Any]],
    ) -> list[str]:
        """Load LoRA weights and patch to the target module's LoRA layers.

        Args:
            lora_args: One LoRA argument dict or a list of LoRA argument dicts.
                lora_id: Unique LoRA model id.
                path: Safetensors file path to load.
                target_module: Pipeline module name to patch. If omitted or
                    None, the pipeline default target module is used.
                scale: Initial LoRA scale. If omitted, 1.0 is used.

        Returns:
            LoRA ids that were successfully loaded.
        """
        if isinstance(lora_args, dict):
            lora_args = [lora_args]
        lora_ids = []
        for args in lora_args:
            lora_id = self._load_lora(args)
            if lora_id is not None:
                lora_ids.append(lora_id)
        return lora_ids

    def _load_lora(self, lora_args: dict[str, Any]) -> str | None:
        lora_ref = LoRARef(**lora_args)
        if lora_ref.target_module is None:
            logger.warning(
                f"LoRA '{lora_ref.lora_id}' has no target module, using default '{self._lora_target_module}'."
            )
            lora_ref.target_module = self._lora_target_module
        if lora_ref.lora_id in self.lora_refs:
            logger.warning(f"LoRA '{lora_ref.lora_id}' is already loaded.")
            return None

        target_module = lora_ref.target_module
        if target_module not in self.lora_layers:
            self.convert_to_lora_layers(target_module)
        layers = self.lora_layers[target_module]

        state_dict = load_safetensors(lora_ref.path)
        lora_weights_dict = convert_lora_state_dict(state_dict)  # layer_name -> LoRAWeights
        for layer_name, lora_weights in lora_weights_dict.items():
            if layer_name not in layers:
                logger.warning(f"No matching layer found for LoRA key '{layer_name}' in module '{target_module}'.")
                continue
            layers[layer_name].load_lora(lora_ref.lora_id, lora_weights, lora_ref.scale)

        lora_ref.status = LoRAStatus.ACTIVE
        self.lora_refs[lora_ref.lora_id] = lora_ref
        logger.info(f"Loaded LoRA '{lora_ref.lora_id}' from '{lora_ref.path}' with scale {lora_ref.scale}.")
        return lora_ref.lora_id

    def unload_loras(self, lora_ids: str | list[str] | None = None):
        """Unload LoRA weights that are not merged.

        Args:
            lora_ids: LoRA id or LoRA ids to unload.
                If None, unload all loaded LoRAs that are not merged.
        """
        if lora_ids is None:
            lora_ids = list(self.lora_refs.keys())
        elif isinstance(lora_ids, str):
            lora_ids = [lora_ids]

        ids_by_module: dict[str, list[str]] = {}
        for lora_id in lora_ids:
            lora_ref = self.lora_refs.get(lora_id)
            if lora_ref is None:
                logger.warning(f"LoRA '{lora_id}' not found.")
                continue
            if lora_ref.status == LoRAStatus.MERGED:
                logger.warning(f"LoRA '{lora_id}' is merged and cannot be unloaded.")
                continue
            ids_by_module.setdefault(lora_ref.target_module, []).append(lora_id)

        for target_module, lora_ids in ids_by_module.items():
            for layer in self.lora_layers.get(target_module, {}).values():
                layer.unload_loras(lora_ids)
            for lora_id in lora_ids:
                del self.lora_refs[lora_id]
            logger.info(f"Unloaded LoRAs {lora_ids} from module '{target_module}'.")

    def set_active_loras(
        self,
        lora_ids: str | list[str],
        scales: float | list[float] | None = None,
    ):
        """Set selected LoRAs active and deactivate other unmerged LoRAs.

        Args:
            lora_ids: LoRA id or LoRA ids to set active.
            scales: Optional scale override for selected LoRAs.
                If float, apply the same scale to every selected LoRA.
                If list[float], apply one scale per LoRA id; its length must
                    match ``lora_ids``.
                If None, keep each selected LoRA's current scale.
        """
        if isinstance(lora_ids, str):
            lora_ids = [lora_ids]

        if isinstance(scales, (int, float)):
            scales = [scales] * len(lora_ids)
        elif scales is None:
            scales = [None] * len(lora_ids)
        if len(lora_ids) != len(scales):
            raise ValueError("lora_ids and scales must have same length.")

        inactive_ids = []
        for lora_id, lora_ref in self.lora_refs.items():
            if lora_id in lora_ids or lora_ref.status == LoRAStatus.MERGED:
                continue
            inactive_ids.append(lora_id)

        self.activate_loras(lora_ids, scales)
        self.deactivate_loras(inactive_ids)

    def activate_loras(
        self,
        lora_ids: str | list[str],
        scales: float | list[float] | None = None,
    ):
        """Activate selected LoRAs without changing other LoRA statuses.

        Args:
            lora_ids: LoRA id or LoRA ids to activate.
            scales: Optional scale override for activated LoRAs.
                If float, apply the same scale to every selected LoRA.
                If list[float], apply one scale per LoRA id; its length must
                    match ``lora_ids``.
                If None, keep each selected LoRA's current scale.
        """
        if isinstance(lora_ids, str):
            lora_ids = [lora_ids]

        if isinstance(scales, (int, float)):
            scales = [scales] * len(lora_ids)
        elif scales is None:
            scales = [None] * len(lora_ids)
        if len(lora_ids) != len(scales):
            raise ValueError("lora_ids and scales must have same length.")

        refs_by_module: dict[str, list[LoRARef]] = {}
        for lora_id, scale in zip(lora_ids, scales):
            lora_ref = self.lora_refs.get(lora_id)
            if lora_ref is None:
                logger.warning(f"LoRA '{lora_id}' not found.")
                continue
            if lora_ref.status == LoRAStatus.MERGED:
                logger.warning(f"LoRA '{lora_id}' is merged and cannot be activated.")
                continue
            if scale is not None:
                lora_ref.scale = scale
            refs_by_module.setdefault(lora_ref.target_module, []).append(lora_ref)

        for target_module, lora_refs in refs_by_module.items():
            lora_ids = [lora_ref.lora_id for lora_ref in lora_refs]
            scales = [lora_ref.scale for lora_ref in lora_refs]
            for layer in self.lora_layers.get(target_module, {}).values():
                layer.activate_loras(lora_ids, scales)
            for lora_ref in lora_refs:
                lora_ref.status = LoRAStatus.ACTIVE
            logger.info(f"Activated LoRAs {lora_ids} on module '{target_module}'.")

    def deactivate_loras(self, lora_ids: str | list[str] | None = None):
        """Deactivate LoRAs while keeping their weights loaded.

        Args:
            lora_ids: LoRA id or LoRA ids to deactivate.
                If None, deactivate all loaded LoRAs.
        """
        if lora_ids is None:
            lora_ids = list(self.lora_refs.keys())
        elif isinstance(lora_ids, str):
            lora_ids = [lora_ids]
        else:
            lora_ids = lora_ids

        ids_by_module: dict[str, list[str]] = {}
        for lora_id in lora_ids:
            lora_ref = self.lora_refs.get(lora_id)
            if lora_ref is None:
                logger.warning(f"LoRA '{lora_id}' not found.")
                continue
            if lora_ref.status != LoRAStatus.ACTIVE:
                logger.warning(f"LoRA '{lora_id}' is not active and cannot be deactivated.")
                continue
            ids_by_module.setdefault(lora_ref.target_module, []).append(lora_id)

        for target_module, lora_ids in ids_by_module.items():
            for layer in self.lora_layers.get(target_module, {}).values():
                layer.deactivate_loras(lora_ids)
            for lora_id in lora_ids:
                self.lora_refs[lora_id].status = LoRAStatus.INACTIVE
            logger.info(f"Deactivated LoRAs {lora_ids} on module '{target_module}'.")

    def merge_loras(
        self,
        target_module: str | None = None,
        chunked: bool = False,
        high_precision: bool = True,
    ):
        """Merge active LoRA weights into base weights.

        Args:
            target_module: Target module to merge.
                If None, merge active LoRAs in all converted target modules.
            chunked: If True, merge in chunks to limit peak memory usage.
            high_precision: If True, compute merge in float32 for better numerical accuracy.
        """
        if target_module is None:
            target_modules = set(self.lora_layers.keys())
        else:
            target_modules = {target_module}

        for target_module in target_modules:
            active_ids = [
                lora_id
                for lora_id, lora_ref in self.lora_refs.items()
                if lora_ref.target_module == target_module and lora_ref.status == LoRAStatus.ACTIVE
            ]
            if not active_ids:
                logger.warning(f"No active LoRAs found in module '{target_module}'.")
                continue
            for layer in self.lora_layers.get(target_module, {}).values():
                layer.merge_loras(chunked=chunked, high_precision=high_precision)
            for lora_id in active_ids:
                self.lora_refs[lora_id].status = LoRAStatus.MERGED
            logger.info(
                f"Merged LoRAs {active_ids} into module '{target_module}' "
                f"(chunked={chunked}, high_precision={high_precision})."
            )

    def unmerge_loras(self, target_module: str | None = None):
        """Undo merged LoRAs and discard their LoRA refs.

        Args:
            target_module: Target module to unmerge.
                If None, unmerge LoRAs in all converted target modules.
        """
        if target_module is None:
            target_modules = set(self.lora_layers.keys())
        else:
            target_modules = {target_module}

        for target_module in target_modules:
            merged_ids = [
                lora_id
                for lora_id, lora_ref in self.lora_refs.items()
                if lora_ref.target_module == target_module and lora_ref.status == LoRAStatus.MERGED
            ]
            if not merged_ids:
                logger.warning(f"No merged LoRAs found in module '{target_module}'.")
                continue
            for layer in self.lora_layers.get(target_module, {}).values():
                layer.unmerge_loras()
            for lora_id in merged_ids:
                del self.lora_refs[lora_id]
            logger.info(f"Unmerged LoRAs {merged_ids} from module '{target_module}'.")

    def reset_loras(self, target_module: str | None = None):
        """Reset LoRA status and restore base weights for selected modules.

        Args:
            target_module: Target module to reset.
                If None, reset LoRA status in all converted target modules.
        """
        if target_module is None:
            target_modules = set(self.lora_layers.keys())
        else:
            target_modules = {target_module}

        for target_module in target_modules:
            lora_ids = [
                lora_id for lora_id, lora_ref in self.lora_refs.items() if lora_ref.target_module == target_module
            ]
            if not lora_ids:
                logger.warning(f"No LoRAs found in module '{target_module}'.")
                continue
            for layer in self.lora_layers.get(target_module, {}).values():
                layer.reset_loras()
            for lora_id in lora_ids:
                del self.lora_refs[lora_id]
            logger.info(f"Reset LoRAs {lora_ids} in module '{target_module}'.")

    def list_loras(self, lora_ids: str | list[str] | None = None) -> list[dict[str, Any]]:
        """List loaded LoRAs and their current status.

        Args:
            lora_ids: LoRA id or LoRA ids to list.
                If None, list all loaded LoRAs.

        Returns:
            List of dicts with keys: lora_id, path, target_module, scale, status.
        """
        if lora_ids is None:
            lora_ids = list(self.lora_refs.keys())
        elif isinstance(lora_ids, str):
            lora_ids = [lora_ids]
        else:
            lora_ids = lora_ids

        result = []
        for lora_id in lora_ids:
            lora_ref = self.lora_refs.get(lora_id)
            if lora_ref is None:
                logger.warning(f"LoRA '{lora_id}' not found.")
                continue
            result.append(
                {
                    "lora_id": lora_ref.lora_id,
                    "path": lora_ref.path,
                    "target_module": lora_ref.target_module,
                    "scale": lora_ref.scale,
                    "status": lora_ref.status.value if lora_ref.status else None,
                }
            )
        return result
