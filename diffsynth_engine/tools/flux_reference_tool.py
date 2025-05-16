from diffsynth_engine import ControlNetParams, FluxImagePipeline, FluxIPAdapter, FluxRedux, fetch_model
from typing import List, Tuple, Optional
from PIL import Image
import torch


class FluxReduxRefTool:
    """
    Use this tool to generate images with reference image based FluxRedux
    """

    def __init__(
        self,
        flux_model_path: str,
        lora_list: List[Tuple[str, float]] = [],
        load_text_encoder=True,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        offload_mode: Optional[str] = None,
    ):
        self.pipe: FluxImagePipeline = FluxImagePipeline.from_pretrained(
            flux_model_path, load_text_encoder=load_text_encoder, device=device, offload_mode=offload_mode
        )
        self.pipe.load_loras(lora_list)
        redux_model_path = fetch_model("muse/flux1-redux-dev", path="flux1-redux-dev.safetensors", revision="v1")
        flux_redux: FluxRedux = FluxRedux.from_pretrained(redux_model_path, device=device)
        self.pipe.load_redux(flux_redux)

    def __call__(
        self,
        ref_image: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        ref_scale: float = 1.0,
        seed: int = 42,
        num_inference_steps: int = 50,
        controlnet_params: List[ControlNetParams] = [],
    ):
        self.pipe.redux.set_scale(ref_scale)
        return self.pipe(
            ref_image=ref_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            controlnet_params=controlnet_params,
            flux_guidance_scale=2.5,
        )


class FluxIPAdapterRefTool:
    """
    Use this tool to generate images with reference image based IP-Adapter
    """

    def __init__(
        self,
        flux_model_path: str,
        lora_list: List[Tuple[str, float]] = [],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        offload_mode: Optional[str] = None,
    ):
        self.pipe: FluxImagePipeline = FluxImagePipeline.from_pretrained(
            flux_model_path, device=device, offload_mode=offload_mode
        )
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
            controlnet_params=controlnet_params,
        )
