import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

_EXTENSION_NAME = "qwen_image_cuda_ext_v3"


def _sources() -> list[str]:
    base_dir = Path(__file__).resolve().parent / "csrc"
    return [
        str(base_dir / "qwen_image_rotary_binding.cpp"),
        str(base_dir / "qwen_image_rotary_kernel.cu"),
    ]


def _extension_arch_list() -> str:
    # Build for the local visible GPU arch directly to avoid PTX JIT/toolchain
    # mismatches, while also bypassing problematic global TORCH_CUDA_ARCH_LIST.
    try:
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:
        return "9.0"
    return f"{major}.{minor}"


def _preferred_host_compilers() -> tuple[Optional[str], Optional[str]]:
    # Prefer system GCC/G++ over conda-forge GCC to avoid nvcc host-compiler
    # compatibility errors in some conda environments.
    cc = "/usr/bin/gcc" if Path("/usr/bin/gcc").exists() else shutil.which("gcc")
    cxx = "/usr/bin/g++" if Path("/usr/bin/g++").exists() else shutil.which("g++")
    return cc, cxx


@lru_cache(maxsize=1)
def _load_extension():
    if not torch.cuda.is_available():
        return None

    old_arch = os.getenv("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = _extension_arch_list()
    old_cc = os.getenv("CC")
    old_cxx = os.getenv("CXX")
    cc, cxx = _preferred_host_compilers()
    if cc:
        os.environ["CC"] = cc
    if cxx:
        os.environ["CXX"] = cxx

    try:
        return load(
            name=_EXTENSION_NAME,
            sources=_sources(),
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=os.getenv("QWEN_IMAGE_CUDA_EXT_VERBOSE", "0") == "1",
        )
    except Exception as err:
        if os.getenv("QWEN_IMAGE_CUDA_EXT_WARN", "1") == "1":
            print(f"[QwenImage CUDA] rotary extension disabled: {err}")
        return None
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch
        if old_cc is None:
            os.environ.pop("CC", None)
        else:
            os.environ["CC"] = old_cc
        if old_cxx is None:
            os.environ.pop("CXX", None)
        else:
            os.environ["CXX"] = old_cxx


def rotary_emb_forward(x: torch.Tensor, freqs_cis: torch.Tensor) -> Optional[torch.Tensor]:
    ext = _load_extension()
    if ext is None:
        return None
    return ext.rotary_emb_forward(x, freqs_cis)


def modulate_forward(x: torch.Tensor, mod_params: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    ext = _load_extension()
    if ext is None:
        return None
    return ext.modulate_forward(x, mod_params)


def modulate_indexed_forward(
    x: torch.Tensor, mod_params: torch.Tensor, index: torch.Tensor
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    ext = _load_extension()
    if ext is None:
        return None
    return ext.modulate_indexed_forward(x, mod_params, index)
