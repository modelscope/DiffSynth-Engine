import torch
import torch.nn as nn

from diffsynth_engine.utils import logging
from diffsynth_engine.utils.platform import pin_memory

logger = logging.get_logger(__name__)


LORA_MERGE_CHUNK_BYTES = 32 * 1024 * 1024


class LoRAWeights(nn.Module):
    """LoRA model weights for one linear layer.

    Attributes:
        down: Down-projection weight with shape ``(rank, in_features)``.
        up: Up-projection weight with shape ``(out_features, rank)``.
        rank: LoRA rank.
        alpha: LoRA alpha used by ``scaling``.
        scale: Current scale used by ``scaling``.
        active: Whether this LoRA model contributes to the layer forward pass.
    """

    def __init__(
        self,
        down: torch.Tensor,
        up: torch.Tensor,
        rank: int,
        alpha: float,
        scale: float = 1.0,
        active: bool = True,
    ):
        super().__init__()
        self.down = nn.Parameter(down, requires_grad=False)
        self.up = nn.Parameter(up, requires_grad=False)
        self.rank = rank
        self.alpha = alpha
        self.scale = scale
        self.active = active

    @property
    def scaling(self) -> float:
        return self.scale * (self.alpha / self.rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA output: x @ down^T @ up^T * scaling.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            LoRA delta tensor of shape (..., out_features).
        """
        return (x @ self.down.t() @ self.up.t()) * self.scaling

    def merge_into(self, weight: torch.Tensor, chunk_bytes: int = 0, high_precision: bool = True):
        """Merge LoRA weights into the base layer weight in-place.

        Args:
            weight: Base layer weight tensor to merge into, shape (out_features, in_features).
            chunk_bytes: If > 0, merge in chunks of this size to limit memory usage.
            high_precision: If True, compute merge in float32 for better numerical accuracy.
        """
        up = self.up.float() if high_precision else self.up
        down = self.down.float() if high_precision else self.down

        if chunk_bytes <= 0:
            if high_precision:
                weight.copy_((weight.float().addmm_(up, down, alpha=self.scaling)).to(weight.dtype))
            else:
                weight.add_(up @ down, alpha=self.scaling)
            return

        chunk_rows = max(1, chunk_bytes // (weight.element_size() * weight.shape[-1]))
        out_dim = up.shape[0]

        for start in range(0, out_dim, chunk_rows):
            end = min(start + chunk_rows, out_dim)
            if high_precision:
                chunk = weight[start:end].float()
                chunk.addmm_(up[start:end], down, alpha=self.scaling)
                weight[start:end] = chunk.to(weight.dtype)
            else:
                weight[start:end].add_(up[start:end] @ down, alpha=self.scaling)


class LinearWithLoRA(nn.Module):
    """Linear layer wrapper that supports multiple LoRAs."""

    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.base_layer = base_layer
        self.lora_weights_dict: dict[str, LoRAWeights] = {}  # lora_id -> LoRAWeights
        self.merged: bool = False
        self._original_weight: torch.Tensor | None = None

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return getattr(self.base_layer, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base linear + sum of active LoRA deltas.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        out = self.base_layer(x)
        for weights in self.lora_weights_dict.values():
            if weights.active:
                out = out + weights(x)
        return out

    def load_lora(
        self,
        lora_id: str,
        lora_weights: LoRAWeights,
        scale: float,
    ):
        """Load LoRA model weights on this layer.

        Args:
            lora_id: Unique id for each LoRA.
            lora_weights: Per-LoRA weights.
            scale: Per-LoRA scaling factors.
        """
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        lora_weights.scale = scale
        lora_weights = lora_weights.to(device=device, dtype=dtype, non_blocking=True)
        self.lora_weights_dict[lora_id] = lora_weights

    def unload_loras(self, lora_ids: list[str]):
        """Unload LoRA weights from this layer.

        Args:
            lora_ids: LoRA ids to unload. Ids not present on this layer are skipped.
        """
        for lora_id in lora_ids:
            if lora_id not in self.lora_weights_dict:
                continue
            del self.lora_weights_dict[lora_id]

    def activate_loras(self, lora_ids: list[str], scales: list[float | None]):
        """Activate LoRA models on this layer.

        Args:
            lora_ids: LoRA ids to activate. Ids not present on this layer are skipped.
            scales: Scale overrides aligned with ``lora_ids``.
                If an item is float, update the corresponding LoRA scale.
                If an item is None, keep the corresponding LoRA's existing scale.
        """
        for lora_id, scale in zip(lora_ids, scales):
            lora_weights = self.lora_weights_dict.get(lora_id)
            if lora_weights is None:
                continue
            lora_weights.active = True
            if scale is not None:
                lora_weights.scale = scale

    def deactivate_loras(self, lora_ids: list[str]):
        """Deactivate LoRA models on this layer.

        Args:
            lora_ids: LoRA ids to deactivate. Ids not present on this layer are skipped.
        """
        for lora_id in lora_ids:
            lora_weights = self.lora_weights_dict.get(lora_id)
            if lora_weights is None:
                continue
            lora_weights.active = False

    def _save_original_weight(self):
        if self._original_weight is not None:
            return
        weight = self.base_layer.weight.data
        self._original_weight = pin_memory(weight.detach().cpu().clone())

    def merge_loras(self, chunked: bool = False, high_precision: bool = True) -> list[str]:
        """Merge active LoRA models into the wrapped base layer weight.

        Args:
            chunked: If True, merge in chunks to limit peak memory usage.
            high_precision: If True, compute merge in float32 for better numerical accuracy.

        Returns:
            LoRA ids that were merged on this layer.
        """
        ids_to_merge = [lora_id for lora_id, lora_weights in self.lora_weights_dict.items() if lora_weights.active]
        if not ids_to_merge:
            return []

        self._save_original_weight()
        chunk_bytes = LORA_MERGE_CHUNK_BYTES if chunked else 0
        for lora_id in ids_to_merge:
            self.lora_weights_dict[lora_id].merge_into(
                self.base_layer.weight.data, chunk_bytes=chunk_bytes, high_precision=high_precision
            )
            del self.lora_weights_dict[lora_id]
        self.merged = True
        return ids_to_merge

    def unmerge_loras(self):
        """Restore the wrapped base weight saved before merge.

        If this layer has no merged LoRAs, this is a no-op.
        """
        if not self.merged:
            return

        self.base_layer.weight.data.copy_(self._original_weight, non_blocking=True)
        self.merged = False

    def reset_loras(self):
        """Reset this layer's LoRA weights and merge status.

        This unmerges the base weight if needed and unloads all LoRA weights
        from this layer.
        """
        self.unmerge_loras()
        self.lora_weights_dict.clear()


LoRALayer = LinearWithLoRA


_lora_layer_mapping: dict[type[nn.Module], type[LoRALayer]] = {
    nn.Linear: LinearWithLoRA,
}


def wrap_with_lora_layer(layer: nn.Module) -> LoRALayer | None:
    """Wrap a supported layer with a LoRA-enabled layer.

    Args:
        layer: A nn.Module to wrap (e.g. nn.Linear).

    Returns:
        LoRA-enabled layer if the layer type is supported; otherwise ``None``.
    """
    for base_type, lora_type in _lora_layer_mapping.items():
        if isinstance(layer, base_type):
            return lora_type(layer)
    return None
