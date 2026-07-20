from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch


@dataclass(frozen=True)
class AttentionRequest:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: Optional[torch.Tensor] = None
    attn_mask: Optional[torch.Tensor] = None
    scale: Optional[float] = None
    kwargs: dict[str, Any] = field(default_factory=dict)


AvailabilityCheck = Callable[[], bool]
SupportCheck = Callable[[AttentionRequest], tuple[bool, str | None]]
AttentionForward = Callable[[AttentionRequest], torch.Tensor]


@dataclass(frozen=True)
class AttentionBackend:
    name: str
    forward: AttentionForward
    devices: frozenset[str] | None = None
    priority: int = 0
    is_available: AvailabilityCheck = lambda: True
    supports: SupportCheck = lambda request: (True, None)
    auto_select: bool = True
    unavailable_reason: str | None = None
    auto_supports: SupportCheck | None = None


_ATTENTION_BACKENDS: dict[str, AttentionBackend] = {}


def register_attention_backend(backend: AttentionBackend, *, overwrite: bool = False) -> None:
    if backend.name in _ATTENTION_BACKENDS and not overwrite:
        raise ValueError(f"Attention backend {backend.name!r} is already registered")
    _ATTENTION_BACKENDS[backend.name] = backend


def get_attention_backend(name: str) -> AttentionBackend:
    try:
        return _ATTENTION_BACKENDS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ATTENTION_BACKENDS))
        raise ValueError(f"Invalid attention implementation {name!r}. Available backends: {available}") from exc


def _check_backend(
    backend: AttentionBackend,
    request: AttentionRequest,
    *,
    auto: bool = False,
) -> tuple[bool, str | None]:
    device_type = request.q.device.type
    if backend.devices is not None and device_type not in backend.devices:
        return False, f"backend {backend.name!r} does not support device type {device_type!r}"
    if not backend.is_available():
        reason = backend.unavailable_reason or f"backend {backend.name!r} is not available in the current environment"
        return False, reason
    support_check = backend.auto_supports if auto and backend.auto_supports is not None else backend.supports
    return support_check(request)


def resolve_attention_backend(name: str, request: AttentionRequest) -> AttentionBackend:
    if name not in {"auto", None}:
        backend = get_attention_backend(name)
        supported, reason = _check_backend(backend, request)
        if not supported:
            raise RuntimeError(reason or f"Attention backend {name!r} does not support this request")
        return backend

    candidates = sorted(
        (backend for backend in _ATTENTION_BACKENDS.values() if backend.auto_select),
        key=lambda backend: backend.priority,
        reverse=True,
    )
    failures = []
    for backend in candidates:
        supported, reason = _check_backend(backend, request, auto=True)
        if supported:
            return backend
        if reason:
            failures.append(reason)
    raise RuntimeError("No attention backend supports this request: " + "; ".join(failures))


def list_attention_backends() -> tuple[str, ...]:
    return tuple(sorted(_ATTENTION_BACKENDS))
