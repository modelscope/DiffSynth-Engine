"""Platform registration and initialization tests without accelerator hardware."""

import subprocess
import sys
from unittest.mock import Mock, patch

import pytest
import torch

import diffsynth_engine.platforms as platforms
import diffsynth_engine.registry as registry
from diffsynth_engine.configs import PipelineConfig
from diffsynth_engine.platforms.ascend import AscendPlatform


@pytest.fixture(autouse=True)
def reset_platform(monkeypatch):
    monkeypatch.setattr(platforms, "_platform", None)
    platforms.auto_detect_device.cache_clear()
    yield
    platforms.auto_detect_device.cache_clear()


def test_import_and_cpu_config_do_not_load_ascend():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import diffsynth_engine.platforms as p; "
            "assert p._platform is None; "
            "from diffsynth_engine.configs import PipelineConfig; "
            "PipelineConfig(model_path='test', device='cpu'); "
            "assert p._platform is None; "
            "assert 'diffsynth_engine.platforms.ascend' not in sys.modules; "
            "assert 'torch_npu' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_platform_plugin_registration_is_lazy(monkeypatch):
    monkeypatch.setattr(registry, "PLATFORM_REGISTRY", {})
    monkeypatch.setattr(registry, "_platform_registry_initialized", False)
    plugin = Mock(side_effect=lambda: registry.register_platform("custom", "diffsynth_engine.platforms:CPUPlatform"))
    monkeypatch.setattr(registry, "load_general_plugins", plugin)
    with patch.object(platforms.CPUPlatform, "is_available") as available:
        assert registry.get_platform_class("custom") is platforms.CPUPlatform
        registry.register_platform("custom", "missing.module:MissingClass")
        assert registry.get_platform_class("custom") is platforms.CPUPlatform
        available.assert_not_called()
        plugin.assert_called_once()
    assert platforms._platform is None
    with pytest.raises(ValueError, match="Unsupported device type"):
        registry.get_platform_class("missing")


@pytest.mark.parametrize(
    "cuda,npu,mps,expected",
    [(True, True, True, "cuda"), (False, True, True, "npu"), (False, False, True, "mps"), (False, False, False, "cpu")],
)
def test_auto_detection_order(cuda, npu, mps, expected):
    with (
        patch.object(platforms.CUDAPlatform, "is_available", return_value=cuda),
        patch.object(AscendPlatform, "is_available", return_value=npu),
        patch.object(platforms.MPSPlatform, "is_available", return_value=mps),
    ):
        assert platforms.auto_detect_device() == expected
    assert platforms._platform is None


def test_explicit_unavailable_npu_does_not_fall_back():
    with (
        patch.object(platforms.CUDAPlatform, "is_available", return_value=True),
        patch.object(AscendPlatform, "is_available", return_value=False),
        pytest.raises(RuntimeError, match="'npu' is not available"),
    ):
        platforms.set_platform("npu")
    assert platforms._platform is None


def test_explicit_npu_overrides_auto_detection():
    with (
        patch.object(platforms.CUDAPlatform, "is_available", return_value=True),
        patch.object(AscendPlatform, "is_available", return_value=True),
        patch.object(AscendPlatform, "normalize_device") as normalize,
    ):
        platforms.set_platform("npu:1")
        assert platforms._platform is AscendPlatform
        normalize.assert_not_called()
        assert platforms.get_device_type("auto") == "cuda"
        with pytest.raises(RuntimeError, match="cannot switch to"):
            platforms.set_platform("auto")
    with pytest.raises(RuntimeError, match="cannot switch to"):
        platforms.set_platform("cpu")


def test_config_validates_device_before_attention_without_initializing():
    def check_attention(_):
        available.assert_called_once_with()
        normalize.assert_called_once_with("cpu")
        assert platforms._platform is None

    with (
        patch.object(platforms.CPUPlatform, "is_available", return_value=True) as available,
        patch.object(
            platforms.CPUPlatform, "normalize_device", wraps=platforms.CPUPlatform.normalize_device
        ) as normalize,
        patch.object(platforms.CPUPlatform, "set_device") as set_device,
        patch("diffsynth_engine.configs.base.get_attn_backend", side_effect=check_attention) as get_attention,
    ):
        config = PipelineConfig(model_path="test", device="cpu")
        assert config.device == "cpu"
        get_attention.assert_called_once_with("sdpa")
        set_device.assert_not_called()
    assert platforms._platform is None


@pytest.mark.parametrize("device", ["auto", "cuda:1", torch.device("cuda:1")])
def test_config_resolves_device_without_selecting_platform(device):
    with (
        patch.object(platforms.CUDAPlatform, "is_available", return_value=True),
        patch.object(platforms.CUDAPlatform, "device_count", return_value=2),
    ):
        config = PipelineConfig(model_path="test", device=device)
        assert config.device == ("cuda" if device == "auto" else str(device))
        PipelineConfig(model_path="test", device="cpu")
    assert platforms._platform is None


def test_config_preserves_backend_normalization():
    with patch.object(platforms.CPUPlatform, "normalize_device", return_value=torch.device("cpu:0")):
        config = PipelineConfig(model_path="test", device="cpu")
    assert config.device == "cpu:0"
    assert platforms._platform is None


@pytest.mark.parametrize("device,index", [("cuda:0", 0), ("cuda:1", 1)])
def test_config_checks_accelerator_index(device, index):
    with (
        patch.object(platforms.CUDAPlatform, "is_available", return_value=True),
        patch.object(platforms.CUDAPlatform, "device_count", return_value=1),
    ):
        if index == 0:
            assert PipelineConfig(model_path="test", device=device).device == device
        else:
            with pytest.raises(ValueError, match="out of range"):
                PipelineConfig(model_path="test", device=device)


def test_config_checks_normalized_index():
    with (
        patch.object(platforms.CUDAPlatform, "is_available", return_value=True),
        patch.object(platforms.CUDAPlatform, "device_count", return_value=1),
        patch.object(platforms.CUDAPlatform, "normalize_device", return_value=torch.device("cuda:1")),
        pytest.raises(ValueError, match="cuda:1.*out of range"),
    ):
        PipelineConfig(model_path="test", device="cuda")
    assert platforms._platform is None


def test_config_does_not_treat_cpu_index_as_accelerator_index():
    with patch.object(platforms.CPUPlatform, "device_count") as device_count:
        assert PipelineConfig(model_path="test", device="cpu:99").device == "cpu:99"
        device_count.assert_not_called()


@pytest.mark.parametrize(
    "device,error,match",
    [
        ("npu", RuntimeError, "not available"),
        ("unknown", ValueError, "Unsupported"),
        ("cpu:invalid", RuntimeError, "device"),
    ],
)
def test_config_rejects_invalid_device(device, error, match):
    with patch.object(AscendPlatform, "is_available", return_value=False), pytest.raises(error, match=match):
        PipelineConfig(model_path="test", device=device)
    assert platforms._platform is None


def test_config_still_rejects_unavailable_attention():
    with (
        patch("diffsynth_engine.configs.base.get_attn_backend", side_effect=RuntimeError("attention unavailable")),
        pytest.raises(RuntimeError, match="attention unavailable"),
    ):
        PipelineConfig(model_path="test", device="cpu")
    assert platforms._platform is None


def test_repeated_platform_initialization_does_not_bind_device():
    with (
        patch.object(platforms.CUDAPlatform, "is_available", return_value=True),
        patch.object(platforms.CUDAPlatform, "set_device") as set_device,
    ):
        platforms.set_platform("cuda:0")
        assert platforms._platform is platforms.CUDAPlatform
        platforms.set_platform("cuda:1")
        assert platforms._platform is platforms.CUDAPlatform
        platforms.set_platform("auto")
        assert platforms._platform is platforms.CUDAPlatform
        set_device.assert_not_called()


def test_local_engine_initializes_before_loading_plugin_pipeline():
    from diffsynth_engine.engine import LocalEngine

    config = PipelineConfig(model_path="test", device="cpu", pipeline_class_name="CustomPipeline")
    pipeline = Mock()

    def load(config):
        assert platforms.get_platform() is platforms.CPUPlatform
        return pipeline

    with patch("diffsynth_engine.engine.get_pipeline_class") as get_pipeline:
        get_pipeline.return_value.from_pretrained.side_effect = load
        engine = LocalEngine(config)
        assert engine.pipeline is pipeline
        engine.shutdown()


def test_distributed_parent_initializes_and_binds_platform():
    from diffsynth_engine.engine import DistributedEngine

    config = PipelineConfig(model_path="test", device="cpu", parallelism=2)
    parent_conn, child_conn = Mock(), Mock()
    parent_conn.recv.return_value = {"status": "ready"}
    with (
        patch("diffsynth_engine.engine.mp.get_context") as get_context,
        patch.object(platforms.CPUPlatform, "set_device") as set_device,
    ):
        context = get_context.return_value
        context.Pipe.return_value = (parent_conn, child_conn)
        engine = DistributedEngine(config, num_workers=2)
        assert platforms.get_platform() is platforms.CPUPlatform
        set_device.assert_called_once_with(0)
        get_context.assert_called_once_with("spawn")
        assert context.Process.return_value.start.call_count == 2
        engine.shutdown()


def test_worker_initializes_platform_before_distributed(monkeypatch):
    import diffsynth_engine.worker as worker

    config = PipelineConfig(model_path="test", device="cpu")
    assert platforms._platform is None
    for name in ("MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK", "RANK", "WORLD_SIZE"):
        monkeypatch.setenv(name, "")

    def check_distributed(**kwargs):
        assert platforms.get_platform() is platforms.CPUPlatform

    with (
        patch.object(worker, "init_distributed_environment", side_effect=check_distributed) as init_distributed,
        patch.object(worker, "initialize_model_parallel"),
        patch.object(worker, "get_pipeline_class"),
    ):
        worker.Worker(0, 0, 1, "localhost", 29500, config)
        init_distributed.assert_called_once()


def test_fp8_requires_initialization():
    with pytest.raises(RuntimeError, match="Platform must be initialized"):
        platforms.get_fp8_dtype()


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e4m3fnuz])
def test_fp8_uses_selected_platform(dtype):
    platforms.set_platform("cpu")
    with patch.object(platforms.CPUPlatform, "fp8_dtype", return_value=dtype):
        assert platforms.get_fp8_dtype() is dtype


@pytest.mark.parametrize("state", ["available", "unavailable", "execution_error"])
def test_ascend_device_detection_checks_and_caches_runtime(state):
    from diffsynth_engine.platforms import ascend

    torch_npu = Mock()
    torch_npu.npu.is_available.return_value = state != "unavailable"
    if state == "execution_error":
        torch_npu.npu.synchronize.side_effect = RuntimeError("execution failed")
    AscendPlatform.is_available.cache_clear()
    ascend._probe_npu_runtime.cache_clear()
    try:
        with (
            patch.object(ascend, "_import_torch_npu", return_value=torch_npu),
            patch.object(torch, "zeros") as zeros,
        ):
            assert AscendPlatform.is_available() is (state == "available")
            assert AscendPlatform.is_available() is (state == "available")
            torch_npu.npu.is_available.assert_called_once()
            if state == "unavailable":
                zeros.assert_not_called()
            else:
                torch_npu.npu.synchronize.assert_called_once()
    finally:
        AscendPlatform.is_available.cache_clear()
        ascend._probe_npu_runtime.cache_clear()


def test_auto_detection_caches_result():
    with patch.object(platforms.CUDAPlatform, "is_available", return_value=True) as available:
        assert platforms.auto_detect_device() == "cuda"
        assert platforms.auto_detect_device() == "cuda"
        available.assert_called_once_with()
