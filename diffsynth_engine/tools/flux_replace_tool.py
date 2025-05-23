from diffsynth_engine import ControlNetParams, FluxControlNet, FluxImagePipeline, FluxRedux, fetch_model
from typing import List, Tuple, Optional
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
        lora_list: List[Tuple[str, float]] = [],
        load_text_encoder=True,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        offload_mode: Optional[str] = None,
    ):
        self.pipe: FluxImagePipeline = FluxImagePipeline.from_pretrained(
            flux_model_path, load_text_encoder=load_text_encoder, device=device, offload_mode=offload_mode, dtype=dtype
        )
        self.pipe.load_loras(lora_list)
        redux_model_path = fetch_model("muse/flux1-redux-dev", path="flux1-redux-dev.safetensors", revision="v1")
        flux_redux: FluxRedux = FluxRedux.from_pretrained(redux_model_path, device=device)
        self.pipe.load_redux(flux_redux)
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
        ref_image: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        inpainting_scale: float = 0.9,
        ref_scale: float = 1.0,
        seed: int = 42,
        num_inference_steps: int = 20,
    ):
        assert image.size == mask.size
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
        )
        return result.resize((width, height))
