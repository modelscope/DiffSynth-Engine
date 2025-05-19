from diffsynth_engine import fetch_model, FluxControlNet, ControlNetParams, FluxImagePipeline
from typing import List, Tuple, Optional
from PIL import Image
import torch


class FluxInpaintingTool:
    def __init__(
        self,
        flux_model_path: str,
        lora_list: List[Tuple[str, float]] = [],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        offload_mode: Optional[str] = None,
    ):
        self.pipe = FluxImagePipeline.from_pretrained(flux_model_path, device=device, offload_mode=offload_mode)
        self.pipe.load_loras(lora_list)
        self.controlnet = FluxControlNet.from_pretrained(
            fetch_model(
                "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", path="diffusion_pytorch_model.safetensors"
            ),
            device=device,
            dtype=torch.bfloat16,
        )

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        inpainting_scale: float = 0.9,
        seed: int = 42,
        num_inference_steps: int = 20,
    ):
        assert image.size == mask.size
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=image.width,
            height=image.height,
            num_inference_steps=num_inference_steps,
            seed=seed,
            controlnet_params=ControlNetParams(
                model=self.controlnet,
                image=image,
                mask=mask,
                scale=inpainting_scale,
            ),
        )
