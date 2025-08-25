import torch
from typing import Callable, Dict, List, Tuple, Optional
from PIL import Image

from diffsynth_engine.configs import FluxPipelineConfig, FluxStateDicts, ControlNetParams
from diffsynth_engine.models.flux import FluxControlNet
from diffsynth_engine.pipelines.flux_image import FluxImagePipeline
from diffsynth_engine.utils.download import fetch_model


class FluxOutpaintingTool:
    def __init__(self, flux_pipe: FluxImagePipeline, controlnet: FluxControlNet):
        self.pipe = flux_pipe
        self.controlnet = controlnet

    @classmethod
    def from_pretrained(
        cls,
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
        flux_pipe = FluxImagePipeline.from_pretrained(config)
        controlnet = FluxControlNet.from_pretrained(
            fetch_model(
                "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", path="diffusion_pytorch_model.safetensors"
            ),
            device=device,
            dtype=torch.bfloat16,
        )
        return cls(flux_pipe, controlnet)

    @classmethod
    def from_state_dict(
        cls,
        flux_state_dicts: FluxStateDicts,
        controlnet_state_dict: Dict[str, torch.Tensor],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        offload_mode: Optional[str] = None,
    ):
        config = FluxPipelineConfig(
            model_path="",
            model_dtype=dtype,
            device=device,
            offload_mode=offload_mode,
        )
        flux_pipe = FluxImagePipeline.from_state_dict(flux_state_dicts, config)
        controlnet = FluxControlNet.from_state_dict(controlnet_state_dict, device, dtype)
        return cls(flux_pipe, controlnet)

    def load_loras(self, lora_list: List[Tuple[str, float]], fused: bool = True, save_original_weight: bool = False):
        self.pipe.load_loras(lora_list, fused, save_original_weight)

    def load_lora(self, lora_path: str, scale: float, fused: bool = True, save_original_weight: bool = False):
        self.pipe.load_lora(lora_path, scale, fused, save_original_weight)

    def unload_loras(self):
        self.pipe.unload_loras()

    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        scaling_factor: float = 2.0,
        inpainting_scale: float = 0.9,
        seed: int = 42,
        num_inference_steps: int = 20,
        progress_callback: Optional[Callable] = None,
    ):
        assert scaling_factor >= 1.0, "scale must be >= 1.0"
        width, height = image.width, image.height
        scaled_width, scaled_height = int(width // scaling_factor), int(height // scaling_factor)
        inner_image = image.resize((scaled_width, scaled_height))
        image = Image.new("RGB", (width, height), color=255)
        image.paste(inner_image, (scaled_width // 2, scaled_height // 2))
        # 生成一张中间是黑色，周围都是白色的方形mask
        mask = Image.new("L", (width, height), color=255)
        mask.paste(Image.new("L", (scaled_width, scaled_height), color=0), (scaled_width // 2, scaled_height // 2))
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
            progress_callback=progress_callback,
        )
