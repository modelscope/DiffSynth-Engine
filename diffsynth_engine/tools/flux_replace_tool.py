from diffsynth_engine import (
    FluxPipelineConfig,
    ControlNetParams,
    FluxControlNet,
    FluxImagePipeline,
    FluxRedux,
    fetch_model,
)
from typing import List, Tuple, Optional, Callable
from PIL import Image
import torch


class FluxReplaceByControlTool:
    """
    FluxReplaceTool is a tool that can replace the content of an image with another image.
    It is based on Redux and InpaintingControlNet.
    """

    def __init__(
        self,
        flux_model_path: str,
        load_text_encoder=True,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        offload_mode: Optional[str] = None,
    ):
        config = FluxPipelineConfig(
            model_path=flux_model_path,
            model_dtype=dtype,
            load_text_encoder=load_text_encoder,
            device=device,
            offload_mode=offload_mode,
        )
        self.pipe: FluxImagePipeline = FluxImagePipeline.from_pretrained(config)
        redux_model_path = fetch_model("muse/flux1-redux-dev", path="flux1-redux-dev.safetensors", revision="v1")
        flux_redux = FluxRedux.from_pretrained(redux_model_path, device=device)
        self.pipe.load_redux(flux_redux)
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
        ref_image: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        inpainting_scale: float = 0.9,
        ref_scale: float = 1.0,
        seed: int = 42,
        num_inference_steps: int = 20,
        progress_callback: Optional[Callable] = None,
    ):
        assert image.size == mask.size
        assert self.pipe.redux is not None
        self.pipe.redux.set_scale(ref_scale)
        width, height = image.width, image.height
        resized_width = (width // 16) * 16
        resized_height = (height // 16) * 16
        image = image.resize((resized_width, resized_height))
        mask = mask.resize((resized_width, resized_height))
        ref_image = ref_image.resize((resized_width, resized_height))
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=image.width,
            height=image.height,
            num_inference_steps=num_inference_steps,
            seed=seed,
            ref_image=ref_image,
            controlnet_params=ControlNetParams(
                model=self.controlnet,
                image=image,
                mask=mask,
                scale=inpainting_scale,
            ),
            progress_callback=progress_callback,
        )
        return result.resize((width, height))
