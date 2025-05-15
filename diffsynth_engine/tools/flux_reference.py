from diffsynth_engine import (
    ControlNetParams, FluxImagePipeline, FluxIPAdapter
)
from typing import List, Tuple, Optional
from PIL import Image
import torch


class FluxReferenceTool:
    def __init__(
        self,
        flux_model_path: str,
        lora_list: List[Tuple[str, float]] = [],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        offload_mode: Optional[str] = None,
    ):
        self.pipe: FluxImagePipeline = FluxImagePipeline.from_pretrained(flux_model_path, device=device, offload_mode=offload_mode)
        self.pipe.load_loras(lora_list)
        ip_adapter_path = fetch_model("muse/FLUX.1-dev-IP-Adapter", path="ip-adapter.safetensors", revision="v1")
        ip_adapter: FluxIPAdapter = FluxIPAdapter.from_pretrained(ip_adapter_path, device=device)
        self.pipe.load_ip_adapter(ip_adapter)

    def __call__(
        self,
        ref_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        ref_scale: float = 0.8,
        seed: int = 42,
        num_inference_steps: int = 20,
        controlnet_params: List[ControlNetParams] = [],
    ):        
        self.pipe.ip_adapter.set_scale(ref_scale)
        return self.pipe(
            ref_image=ref_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            controlnet_params=controlnet_params
        )

