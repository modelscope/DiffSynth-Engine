import torch
from typing import Callable, Dict, List, Tuple, Optional
from PIL import Image

from diffsynth_engine.configs import FluxPipelineConfig, FluxStateDicts, ControlNetParams
from diffsynth_engine.models.flux import FluxRedux, FluxControlNet
from diffsynth_engine.pipelines.flux_image import FluxImagePipeline
from diffsynth_engine.utils.download import fetch_model


class FluxReplaceByControlTool:
    """
    FluxReplaceTool is a tool that can replace the content of an image with another image.
    It is based on Redux and InpaintingControlNet.
    """

    def __init__(self, flux_pipe: FluxImagePipeline, redux: FluxRedux, controlnet: FluxControlNet):
        self.pipe = flux_pipe
        self.pipe.load_redux(redux)
        self.controlnet = controlnet

    @classmethod
    def from_pretrained(
        cls,
        flux_model_path: str,
        load_text_encoder: bool = True,
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
        flux_pipe = FluxImagePipeline.from_pretrained(config)
        redux_model_path = fetch_model("muse/flux1-redux-dev", path="flux1-redux-dev.safetensors", revision="v1")
        redux = FluxRedux.from_pretrained(redux_model_path, device=device, dtype=dtype)
        controlnet = FluxControlNet.from_pretrained(
            fetch_model(
                "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
                path="diffusion_pytorch_model.safetensors",
            ),
            device=device,
            dtype=torch.bfloat16,
        )
        return cls(flux_pipe, redux, controlnet)

    @classmethod
    def from_state_dict(
        cls,
        flux_state_dicts: FluxStateDicts,
        redux_state_dict: Dict[str, torch.Tensor],
        controlnet_state_dict: Dict[str, torch.Tensor],
        load_text_encoder: bool = True,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        offload_mode: Optional[str] = None,
    ):
        config = FluxPipelineConfig(
            model_path="",
            model_dtype=dtype,
            load_text_encoder=load_text_encoder,
            device=device,
            offload_mode=offload_mode,
        )
        flux_pipe = FluxImagePipeline.from_state_dict(flux_state_dicts, config)
        redux = FluxRedux.from_state_dict(redux_state_dict, device=device, dtype=dtype)
        controlnet = FluxControlNet.from_state_dict(controlnet_state_dict, device=device, dtype=dtype)
        return cls(flux_pipe, redux, controlnet)

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
