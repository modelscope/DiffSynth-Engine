# modified from transformers.cache_utils
import torch
from typing import Any, Dict, List, Tuple, Optional


class Cache:
    pass


class DynamicCache(Cache):
    def __init__(self):
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif not self.key_cache[layer_idx].numel():
                # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                # expects shape (b, s, ...)
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=1)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=1)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
