from functools import cache

from diffsynth_engine.layers.attention.backends.abstract import AttentionBackend, AttentionType
from diffsynth_engine.utils.import_utils import LazyImport, is_npu_available

AiterBackend = LazyImport("diffsynth_engine.layers.attention.backends.aiter", "AiterBackend")
AiterFP8Backend = LazyImport("diffsynth_engine.layers.attention.backends.aiter", "AiterFP8Backend")
FlashAttention2Backend = LazyImport("diffsynth_engine.layers.attention.backends.flash_attn_2", "FlashAttention2Backend")
FlashAttention3Backend = LazyImport("diffsynth_engine.layers.attention.backends.flash_attn_3", "FlashAttention3Backend")
FlashAttention3FP8Backend = LazyImport(
    "diffsynth_engine.layers.attention.backends.flash_attn_3", "FlashAttention3FP8Backend"
)
FlashAttention4Backend = LazyImport("diffsynth_engine.layers.attention.backends.flash_attn_4", "FlashAttention4Backend")
SageAttention2Backend = LazyImport("diffsynth_engine.layers.attention.backends.sage_attn_2", "SageAttention2Backend")
SageAttention3Backend = LazyImport("diffsynth_engine.layers.attention.backends.sage_attn_3", "SageAttention3Backend")
SDPABackend = LazyImport("diffsynth_engine.layers.attention.backends.sdpa", "SDPABackend")
SpargeAttentionBackend = LazyImport("diffsynth_engine.layers.attention.backends.sparge_attn", "SpargeAttentionBackend")
MindieAttentionBackend = LazyImport("diffsynth_engine.layers.attention.backends.mindie_attn", "MindieAttentionBackend")

_attention_backends = {
    AttentionType.AITER: AiterBackend,
    AttentionType.AITER_FP8: AiterFP8Backend,
    AttentionType.FA2: FlashAttention2Backend,
    AttentionType.FA3: FlashAttention3Backend,
    AttentionType.FA3_FP8: FlashAttention3FP8Backend,
    AttentionType.FA4: FlashAttention4Backend,
    AttentionType.SAGE2: SageAttention2Backend,
    AttentionType.SAGE3: SageAttention3Backend,
    AttentionType.SDPA: SDPABackend,
    AttentionType.SPARGE: SpargeAttentionBackend,
    AttentionType.MINDIE: MindieAttentionBackend,
}


@cache
def get_attn_backend(head_size: int, attn_type: AttentionType | None = None) -> type["AttentionBackend"]:
    # use SDPA as default
    if attn_type is None:
        attn_type = AttentionType.SDPA

    # NPU auto-switch: use MINDIE when NPU is available
    if is_npu_available():
        attn_type = AttentionType.MINDIE

    selected_backend = _attention_backends[attn_type]
    selected_backend.check_availability()
    if not selected_backend.supports_head_size(head_size):
        raise ValueError(f"Head size {head_size} is not supported by {attn_type}")
    return selected_backend
