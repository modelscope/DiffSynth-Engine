"""MindIE availability checks with mocked NPU runtime execution."""

import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from diffsynth_engine.layers.attention.backends.mindie_attn import MindieAttentionBackend
from diffsynth_engine.platforms import ascend


@pytest.fixture(autouse=True)
def reset_availability_cache():
    checks = (
        ascend.AscendPlatform.is_available,
        ascend._probe_npu_runtime,
        ascend.is_mindie_sd_available,
        ascend.is_mindie_attention_available,
        ascend.is_mindie_compile_available,
    )
    for check in checks:
        check.cache_clear()
    try:
        yield
    finally:
        for check in checks:
            check.cache_clear()


@pytest.mark.parametrize(
    "npu_available,import_error",
    [(False, None), (True, None), (True, ImportError("missing")), (True, OSError("unloadable"))],
)
def test_mindie_dependency_availability(npu_available, import_error):
    with (
        patch.object(ascend.AscendPlatform, "is_available", return_value=npu_available),
        patch.object(ascend, "_import_mindie_sd", side_effect=import_error) as import_mindie,
    ):
        expected = npu_available and import_error is None
        assert ascend.is_mindie_sd_available() is expected
        assert ascend.is_mindie_sd_available() is expected
        assert import_mindie.call_count == int(npu_available)


@pytest.mark.parametrize(
    "check,module_name,entry_name,operation_check",
    [
        (
            ascend.is_mindie_attention_available,
            "mindiesd.layers.flash_attn.attention_forward",
            "attention_forward",
            "_probe_mindie_attention",
        ),
        (
            ascend.is_mindie_compile_available,
            "mindiesd.compilation",
            "MindieSDBackend",
            "_probe_mindie_compile",
        ),
    ],
)
@pytest.mark.parametrize(
    "state", ["no_mindie", "import_error", "missing_entry", "not_callable", "probe_failed", "available"]
)
def test_mindie_entry_availability(check, module_name, entry_name, operation_check, state):
    module = ModuleType(module_name)
    entry = Mock()
    if state in ("available", "probe_failed"):
        setattr(module, entry_name, entry)
    elif state == "not_callable":
        setattr(module, entry_name, None)

    with (
        patch.object(ascend, "is_mindie_sd_available", return_value=state != "no_mindie"),
        patch.object(
            ascend.importlib,
            "import_module",
            return_value=module,
            side_effect=ImportError("missing") if state == "import_error" else None,
        ) as import_module,
        patch.object(ascend, operation_check, return_value=state != "probe_failed") as operation_probe,
    ):
        assert check() is (state == "available")
        assert check() is (state == "available")
        if state == "no_mindie":
            import_module.assert_not_called()
        else:
            import_module.assert_called_once_with(module_name)
        if state in ("available", "probe_failed"):
            operation_probe.assert_called_once_with(entry)
        else:
            operation_probe.assert_not_called()


@pytest.mark.parametrize("available", [False, True])
def test_mindie_attention_backend_checks_entry(available):
    with (
        patch.object(ascend.AscendPlatform, "is_available", return_value=True),
        patch.object(ascend, "is_mindie_attention_available", return_value=available),
    ):
        if available:
            MindieAttentionBackend.check_availability()
        else:
            with pytest.raises(RuntimeError, match="MindIE attention backend is not available"):
                MindieAttentionBackend.check_availability()


def test_unavailable_mindie_compile_uses_default_backend():
    with patch.object(ascend, "is_mindie_compile_available", return_value=False):
        assert ascend.AscendPlatform.compile_kwargs() == {}


@pytest.mark.parametrize("construction_error", [None, RuntimeError("backend initialization failed")])
def test_available_mindie_compile_constructs_backend_without_trial_run(monkeypatch, construction_error):
    module = ModuleType("mindiesd.compilation")
    module.CompilationConfig = Mock()
    module.MindieSDBackend = Mock(side_effect=construction_error)
    monkeypatch.setitem(sys.modules, "mindiesd.compilation", module)
    with (
        patch.object(ascend, "is_mindie_compile_available", return_value=True),
        patch.object(ascend.torch, "compile") as compile_model,
    ):
        if construction_error is None:
            assert ascend.AscendPlatform.compile_kwargs() == {"backend": module.MindieSDBackend.return_value}
        else:
            with pytest.raises(RuntimeError, match="backend initialization failed"):
                ascend.AscendPlatform.compile_kwargs()
        module.MindieSDBackend.assert_called_once_with()
        compile_model.assert_not_called()


@pytest.mark.parametrize("kind", ["attention", "compile"])
@pytest.mark.parametrize("state", ["success", "shape", "nonfinite", "execution", "synchronize"])
def test_mindie_operation_checks(kind, state):
    import torch

    value = torch.ones(1, 128, 8, 128) if kind == "attention" else torch.ones(8, 32)
    output = value.clone()
    if state == "shape":
        output = torch.ones(1)
    elif state == "nonfinite":
        output.fill_(float("nan"))
    operation = Mock(
        return_value=output, side_effect=RuntimeError("execution failed") if state == "execution" else None
    )
    npu = Mock()
    if state == "synchronize":
        npu.npu.synchronize.side_effect = RuntimeError("sync failed")
    with (
        patch.object(ascend.torch, "randn", return_value=value),
        patch.object(ascend, "_import_torch_npu", return_value=npu),
        patch.object(ascend.torch, "compile", return_value=operation),
    ):
        if kind == "attention":
            result = ascend._probe_mindie_attention(operation)
        else:
            result = ascend._probe_mindie_compile(Mock())
        assert result is (state == "success")
        operation.assert_called_once()
