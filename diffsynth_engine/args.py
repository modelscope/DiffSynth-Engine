import argparse
from typing import Any, Dict, Tuple

from diffsynth_engine.configs.base import AttentionParams, SpargeAttentionParams
from diffsynth_engine.layers.attention import AttentionType
from diffsynth_engine.utils.platform import str_to_torch_dtype


def _parse_tuple(value: str) -> Tuple[int, int] | int:
    """Parse tuple string, format: '256,256' or '256'"""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) == 1:
        return int(parts[0])
    elif len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    else:
        raise ValueError(f"Cannot parse tuple: {value}, format should be '256,256' or '256'")


def _parse_attention_params(
    attn_type: str,
    sparge_topk: float | None = None,
) -> AttentionParams | None:
    """Parse attention parameters based on attention type"""
    if attn_type == AttentionType.SPARGE:
        if sparge_topk is not None:
            return SpargeAttentionParams(topk=sparge_topk)
        else:
            return SpargeAttentionParams()
    else:
        return None


def parse_cli_args() -> Dict[str, Any]:
    """Parse command line arguments and return args_dict"""
    parser = argparse.ArgumentParser(description="DiffSynth Engine configuration parameters")

    # Define choices
    dtype_choices = ["float32", "float16", "bfloat16"]

    # Model configuration group
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model-path", type=str, required=True, help="Model path")
    model_group.add_argument(
        "--model-dtype",
        type=str,
        default="bf16",
        choices=dtype_choices,
        help="Model data type (default: bf16)",
    )
    model_group.add_argument(
        "--text-encoder-dtype",
        type=str,
        default="bf16",
        choices=dtype_choices,
        help="Text encoder data type (default: bf16)",
    )
    model_group.add_argument(
        "--vae-dtype",
        type=str,
        default="fp32",
        choices=dtype_choices,
        help="VAE data type (default: fp32)",
    )
    model_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)",
    )
    model_group.add_argument(
        "--pipeline-class-name",
        type=str,
        default=None,
        help="Pipeline class name, if None, will infer from model repo",
    )

    # VAE configuration group
    vae_group = parser.add_argument_group("VAE Configuration")
    vae_group.add_argument(
        "--vae-tiled",
        action="store_true",
        help="Enable VAE tiled processing",
    )
    vae_group.add_argument(
        "--vae-tile-size",
        type=str,
        default="256,256",
        help="VAE tile size, format: 'width,height' or single integer (default: 256,256)",
    )
    vae_group.add_argument(
        "--vae-tile-stride",
        type=str,
        default="192,192",
        help="VAE tile stride, format: 'width,height' or single integer (default: 192,192)",
    )

    # Attention configuration group
    attn_group = parser.add_argument_group("Attention Configuration")
    attn_group.add_argument(
        "--attn-type",
        type=str,
        default="sdpa",
        help="Attention type (default: sdpa)",
    )
    attn_group.add_argument(
        "--sparge-topk",
        type=float,
        default=None,
        help="Sparge attention topk parameter (default: 0.5)",
    )

    # Optimization configuration group
    optimization_group = parser.add_argument_group("Optimization Configuration")
    optimization_group.add_argument(
        "--use-torch-compile",
        action="store_true",
        help="Compile repeated transformer blocks with torch.compile",
    )
    optimization_group.add_argument(
        "--compile-ffn",
        action="store_true",
        help="Compile only FFN (MLP) blocks with torch.compile (finer-grained than --use-torch-compile)",
    )

    # Parallelism configuration group
    parallel_group = parser.add_argument_group("Parallelism Configuration")
    parallel_group.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Total number of inference workers (default: 1)",
    )
    parallel_group.add_argument(
        "--use-cfg-parallel",
        action="store_true",
        help="Use CFG parallel",
    )
    parallel_group.add_argument(
        "--sp-ulysses-degree",
        type=int,
        default=None,
        help="Sequence parallel Ulysses degree",
    )
    parallel_group.add_argument(
        "--sp-ring-degree",
        type=int,
        default=None,
        help="Sequence parallel Ring degree",
    )
    parallel_group.add_argument(
        "--tp-degree",
        type=int,
        default=None,
        help="Tensor parallel degree",
    )
    parallel_group.add_argument(
        "--use-vae-parallel",
        action="store_true",
        help="Use VAE parallel (tile-based parallel encode/decode)",
    )
    parallel_group.add_argument(
        "--use-fsdp",
        action="store_true",
        help="Use FSDP (Fully Sharded Data Parallel)",
    )

    args = parser.parse_args()

    args_dict: Dict[str, Any] = {}

    # Model configuration
    args_dict["model_path"] = args.model_path
    args_dict["model_dtype"] = str_to_torch_dtype(args.model_dtype)
    args_dict["text_encoder_dtype"] = str_to_torch_dtype(args.text_encoder_dtype)
    args_dict["vae_dtype"] = str_to_torch_dtype(args.vae_dtype)
    args_dict["device"] = args.device
    args_dict["pipeline_class_name"] = args.pipeline_class_name

    # VAE configuration
    args_dict["vae_tiled"] = args.vae_tiled
    args_dict["vae_tile_size"] = _parse_tuple(args.vae_tile_size)
    args_dict["vae_tile_stride"] = _parse_tuple(args.vae_tile_stride)

    # Attention configuration
    attn_type = args.attn_type.lower()
    args_dict["attn_type"] = attn_type
    args_dict["attn_params"] = _parse_attention_params(attn_type, args.sparge_topk)

    # Optimization configuration
    args_dict["use_torch_compile"] = args.use_torch_compile
    args_dict["compile_ffn"] = args.compile_ffn

    # Parallelism configuration
    args_dict["parallelism"] = args.parallelism
    args_dict["use_cfg_parallel"] = args.use_cfg_parallel
    args_dict["sp_ulysses_degree"] = args.sp_ulysses_degree
    args_dict["sp_ring_degree"] = args.sp_ring_degree
    args_dict["tp_degree"] = args.tp_degree
    args_dict["use_vae_parallel"] = args.use_vae_parallel
    args_dict["use_fsdp"] = args.use_fsdp

    return args_dict


if __name__ == "__main__":
    args_dict = parse_cli_args()
    print(args_dict)
