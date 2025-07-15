from diffsynth_engine import fetch_model, FluxPipelineConfig, FluxControlNet, ControlNetParams, FluxImagePipeline
from typing import List, Tuple, Optional, Callable
from PIL import Image
import torch


class FluxInpaintingTool:
    def __init__(
        self,
        flux_model_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        offload_mode: Optional[str] = None,
    ):
        config = FluxPipelineConfig(
            model_path=flux_model_path,
            model_dtype=dtype,
            device=device,
            offload_mode=offload_mode,
        )
        self.pipe = FluxImagePipeline.from_pretrained(config)
        self.controlnet = FluxControlNet.from_pretrained(
            fetch_model(
                "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", path="diffusion_pytorch_model.safetensors"
            ),
            device=device,
            dtype=torch.bfloat16,
        )

    def load_loras(self, lora_list: List[Tuple[str, float]], fused: bool = True, save_original_weight: bool = False):
        self.pipe.load_loras(lora_list, fused, save_original_weight)

    def load_lora(self, lora_path: str, scale: float, fused: bool = True, save_original_weight: bool = False):
        self.pipe.load_lora(lora_path, scale, fused, save_original_weight)

    def unload_loras(self):
        self.pipe.unload_loras()

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        inpainting_scale: float = 0.9,
        seed: int = 42,
        num_inference_steps: int = 20,
        progress_callback: Optional[Callable] = None,  # def progress_callback(current, total, status)
        controlnet_params: Optional[List[ControlNetParams]] = None,
    ):
        assert image.size == mask.size
        final_controlnet_params = [
            ControlNetParams(
                model=self.controlnet,
                image=image,
                mask=mask,
                scale=inpainting_scale,
            )
        ]
        if controlnet_params is not None:
            final_controlnet_params.extend(controlnet_params)

        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=image.width,
            height=image.height,
            num_inference_steps=num_inference_steps,
            seed=seed,
            controlnet_params=final_controlnet_params,
            progress_callback=progress_callback,
        )
