import os
import torch
import numpy as np
from einops import rearrange
from typing import Dict, List, Tuple, Union, Optional
from PIL import Image

from diffsynth_engine.configs import (
    BaseConfig,
    BaseStateDicts,
    LoraConfig,
    AttnImpl,
    SpargeAttentionParams,
    VideoSparseAttentionParams,
)
from diffsynth_engine.models.basic.video_sparse_attention import get_vsa_kwargs
from diffsynth_engine.utils.offload import enable_sequential_cpu_offload, offload_model_to_dict, restore_model_from_dict
from diffsynth_engine.utils.fp8_linear import enable_fp8_autocast
from diffsynth_engine.utils.gguf import load_gguf_checkpoint
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.loader import load_file
from diffsynth_engine.utils.platform import empty_cache

logger = logging.get_logger(__name__)


class LoRAStateDictConverter:
    def convert(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        return {"lora": lora_state_dict}


class BasePipeline:
    lora_converter = LoRAStateDictConverter()

    def __init__(
        self,
        vae_tiled: bool = False,
        vae_tile_size: int | Tuple[int, int] = -1,
        vae_tile_stride: int | Tuple[int, int] = -1,
        device="cuda",
        dtype=torch.float16,
    ):
        super().__init__()
        self.config = None
        self.vae_tiled = vae_tiled
        self.vae_tile_size = vae_tile_size
        self.vae_tile_stride = vae_tile_stride
        self.device = device
        self.dtype = dtype
        self.offload_mode = None
        self.model_names = []
        self._offload_param_dict = {}
        self.offload_to_disk = False

    @classmethod
    def from_pretrained(cls, model_path_or_config: str | BaseConfig) -> "BasePipeline":
        raise NotImplementedError()

    @classmethod
    def from_state_dict(cls, state_dicts: BaseStateDicts, config: BaseConfig) -> "BasePipeline":
        raise NotImplementedError()

    def update_weights(self, state_dicts: BaseStateDicts) -> None:
        raise NotImplementedError()

    @staticmethod
    def update_component(
        component: torch.nn.Module,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
    ) -> None:
        if component and state_dict:
            component.load_state_dict(state_dict, assign=True)
            component.to(device=device, dtype=dtype, non_blocking=True)

    def _load_lora_state_dicts(
        self,
        lora_state_dict_list: List[Tuple[Dict[str, torch.Tensor], Union[float, LoraConfig], str]],
        fused: bool = True,
        save_original_weight: bool = False,
        lora_converter: Optional[LoRAStateDictConverter] = None,
    ):
        if not lora_converter:
            lora_converter = self.lora_converter

        for state_dict, lora_item, lora_name in lora_state_dict_list:
            if isinstance(lora_item, float):
                lora_scale = lora_item
                scheduler_config = None
            elif isinstance(lora_item, LoraConfig):
                lora_scale = lora_item.scale
                scheduler_config = lora_item.scheduler_config
            else:
                raise ValueError(f"lora_item must be float or LoraConfig, got {type(lora_item)}")

            logger.info(f"loading lora from state_dict '{lora_name}' with scale={lora_scale}")

            if scheduler_config is not None:
                self.apply_scheduler_config(scheduler_config)
                logger.info(f"Applied scheduler args from LoraConfig: {scheduler_config}")

            lora_state_dict = lora_converter.convert(state_dict)
            for model_name, model_state_dict in lora_state_dict.items():
                model = getattr(self, model_name)
                lora_args = []
                for key, param in model_state_dict.items():
                    lora_args.append(
                        {
                            "name": lora_name,
                            "key": key,
                            "scale": lora_scale,
                            "rank": param["rank"],
                            "alpha": param["alpha"],
                            "up": param["up"],
                            "down": param["down"],
                            "device": self.device,
                            "dtype": self.dtype,
                            "save_original_weight": save_original_weight,
                        }
                    )
                model.load_loras(lora_args, fused=fused)

    def load_loras(
        self,
        lora_list: List[Tuple[str, Union[float, LoraConfig]]],
        fused: bool = True,
        save_original_weight: bool = False,
        lora_converter: Optional[LoRAStateDictConverter] = None,
    ):
        lora_state_dict_list = []
        for lora_path, lora_item in lora_list:
            logger.info(f"loading lora from {lora_path}")
            state_dict = load_file(lora_path, device=self.device)
            lora_state_dict_list.append((state_dict, lora_item, lora_path))

        self._load_lora_state_dicts(
            lora_state_dict_list=lora_state_dict_list,
            fused=fused,
            save_original_weight=save_original_weight,
            lora_converter=lora_converter,
        )

    def load_lora(self, path: str, scale: float, fused: bool = True, save_original_weight: bool = False):
        self.load_loras([(path, scale)], fused, save_original_weight)

    def apply_scheduler_config(self, scheduler_config: Dict):
        pass

    def unload_loras(self):
        raise NotImplementedError()

    @staticmethod
    def load_model_checkpoint(
        checkpoint_path: str | List[str],
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
        convert_dtype: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint_path, str):
            checkpoint_path = [checkpoint_path]
        state_dict = {}
        for path in checkpoint_path:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} is not a file")
            elif path.endswith(".safetensors"):
                state_dict_ = load_file(path, device=device)
                if convert_dtype:
                    for key, value in state_dict_.items():
                        state_dict[key] = value.to(dtype)
                else:
                    state_dict.update(state_dict_)

            elif path.endswith(".gguf"):
                state_dict.update(**load_gguf_checkpoint(path, device=device, dtype=dtype))
            else:
                raise ValueError(f"{path} is not a .safetensors or .gguf file")
        return state_dict

    @staticmethod
    def convert(state_dict: Dict[str, torch.Tensor], dtype: torch.dtype):
        for key, value in state_dict.items():
            state_dict[key] = value.to(dtype)
        return state_dict

    @staticmethod
    def validate_image_size(
        height: int,
        width: int,
        minimum: int | None = None,
        maximum: int | None = None,
        multiple_of: int | None = None,
    ):
        if minimum is not None and (height < minimum or width < minimum):
            raise ValueError(f"expects height and width not less than {minimum}")
        if maximum is not None and (height > maximum or width > maximum):
            raise ValueError(f"expects height and width not greater than {maximum}")
        if height % multiple_of != 0 or width % multiple_of != 0:
            raise ValueError(f"expects height and width to be multiples of {multiple_of}")

    @staticmethod
    def preprocess_image(image: Image.Image, mode="RGB") -> torch.Tensor:
        image = image.convert(mode)
        image_array = np.array(image, dtype=np.float32)
        if len(image_array.shape) == 2:
            image_array = image_array[:, :, np.newaxis]
        image = torch.Tensor((image_array / 255) * 2 - 1).permute(2, 0, 1).unsqueeze(0)
        return image

    @staticmethod
    def preprocess_mask(image: Image.Image, mode="L") -> torch.Tensor:
        image = image.convert(mode)
        image_array = np.array(image, dtype=np.float32)
        image = torch.Tensor((image_array / 255)).unsqueeze(0).unsqueeze(0)
        # binary
        image[image < 0.5] = 0
        image[image >= 0.5] = 1
        return image

    @staticmethod
    def preprocess_images(images: List[Image.Image]) -> List[torch.Tensor]:
        return [BasePipeline.preprocess_image(image) for image in images]

    @staticmethod
    def vae_output_to_image(vae_output: torch.Tensor) -> Image.Image | List[Image.Image]:
        vae_output = vae_output[0]
        if vae_output.ndim == 4:
            vae_output = rearrange(vae_output, "c t h w -> t h w c")
        else:
            vae_output = rearrange(vae_output, "c h w -> h w c")

        image = ((vae_output.float() / 2 + 0.5).clip(0, 1) * 255).cpu().numpy().astype("uint8")
        if image.ndim == 4:
            image = [Image.fromarray(img) for img in image]
        else:
            image = Image.fromarray(image)
        return image

    @staticmethod
    def generate_noise(shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise

    def encode_image(
        self, image: torch.Tensor, tiled: bool = False, tile_size: int = 64, tile_stride: int = 32
    ) -> torch.Tensor:
        image = image.to(self.device, self.vae_encoder.dtype)
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_image(self, latent: torch.Tensor) -> torch.Tensor:
        vae_dtype = self.vae_decoder.conv_in.weight.dtype
        latent = latent.to(self.device, vae_dtype)
        image = self.vae_decoder(
            latent, tiled=self.vae_tiled, tile_size=self.vae_tile_size, tile_stride=self.vae_tile_stride
        )
        return image

    def prepare_latents(
        self,
        latents: torch.Tensor,
        input_image: Image.Image,
        denoising_strength: float,
        num_inference_steps: int,
        tiled: bool = False,
        tile_size: int = 64,
        tile_stride: int = 32,
    ):
        # Prepare scheduler
        if input_image is not None:
            total_steps = num_inference_steps
            sigmas, timesteps = self.noise_scheduler.schedule(total_steps)
            t_start = max(total_steps - int(num_inference_steps * denoising_strength), 1)
            sigma_start, sigmas = sigmas[t_start - 1], sigmas[t_start - 1 :]
            timesteps = timesteps[t_start - 1 :]

            self.load_models_to_device(["vae_encoder"])
            noise = latents
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.dtype)
            latents = self.encode_image(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            init_latents = latents.clone()
            latents = self.sampler.add_noise(latents, noise, sigma_start)
        else:
            sigmas, timesteps = self.noise_scheduler.schedule(num_inference_steps)
            # k-diffusion
            # if you have any questions about this, please ask @dizhipeng.dzp for more details
            latents = latents * sigmas[0] / ((sigmas[0] ** 2 + 1) ** 0.5)
            init_latents = latents.clone()
        sigmas, timesteps = (
            sigmas.to(device=self.device, dtype=self.dtype),
            timesteps.to(device=self.device, dtype=self.dtype),
        )
        init_latents, latents = (
            init_latents.to(device=self.device, dtype=self.dtype),
            latents.to(device=self.device, dtype=self.dtype),
        )
        return init_latents, latents, sigmas, timesteps

    def get_attn_kwargs(self, latents: torch.Tensor) -> Dict:
        attn_kwargs = {"attn_impl": self.config.dit_attn_impl.value}
        if isinstance(self.config.attn_params, SpargeAttentionParams):
            assert self.config.dit_attn_impl == AttnImpl.SPARGE
            attn_kwargs.update(
                {
                    "smooth_k": self.config.attn_params.smooth_k,
                    "simthreshd1": self.config.attn_params.simthreshd1,
                    "cdfthreshd": self.config.attn_params.cdfthreshd,
                    "pvthreshd": self.config.attn_params.pvthreshd,
                }
            )
        elif isinstance(self.config.attn_params, VideoSparseAttentionParams):
            assert self.config.dit_attn_impl == AttnImpl.VSA
            attn_kwargs.update(
                get_vsa_kwargs(latents.shape[2:], (1, 2, 2), self.config.attn_params.sparsity, device=self.device)
            )
        return attn_kwargs

    def eval(self):
        for model_name in self.model_names:
            model = getattr(self, model_name)
            if model is not None:
                model.eval()
        return self

    def enable_cpu_offload(self, offload_mode: str | None, offload_to_disk: bool = False):
        valid_offload_mode = ("cpu_offload", "sequential_cpu_offload", "disable", None)
        if offload_mode not in valid_offload_mode:
            raise ValueError(f"offload_mode must be one of {valid_offload_mode}, but got {offload_mode}")
        if self.device == "cpu" or self.device == "mps":
            logger.warning("must set an non cpu device for pipeline before calling enable_cpu_offload")
            return
        if offload_mode is None or offload_mode == "disable":
            self._disable_offload()
        elif offload_mode == "cpu_offload":
            self._enable_model_cpu_offload()
        elif offload_mode == "sequential_cpu_offload":
            self._enable_sequential_cpu_offload()
        self.offload_to_disk = offload_to_disk

    def _enable_model_cpu_offload(self):
        for model_name in self.model_names:
            model = getattr(self, model_name)
            if model is not None:
                self._offload_param_dict[model_name] = offload_model_to_dict(model)
        self.offload_mode = "cpu_offload"

    def _enable_sequential_cpu_offload(self):
        for model_name in self.model_names:
            model = getattr(self, model_name)
            if model is not None:
                enable_sequential_cpu_offload(model, self.device)
        self.offload_mode = "sequential_cpu_offload"

    def _disable_offload(self):
        self.offload_mode = None
        self._offload_param_dict = {}
        for model_name in self.model_names:
            model = getattr(self, model_name)
            if model is not None:
                model.to(self.device)

    def enable_fp8_autocast(
        self, model_names: List[str], compute_dtype: torch.dtype = torch.bfloat16, use_fp8_linear: bool = False
    ):
        for model_name in model_names:
            model = getattr(self, model_name)
            if model is not None:
                model.to(dtype=torch.float8_e4m3fn)
                enable_fp8_autocast(model, compute_dtype, use_fp8_linear)
        self.fp8_autocast_enabled = True

    def load_models_to_device(self, load_model_names: List[str] | None = None):
        load_model_names = load_model_names if load_model_names else []
        # only load models to device if offload_mode is set
        if not self.offload_mode:
            return
        if self.offload_mode == "sequential_cpu_offload":
            # fresh the cuda cache
            empty_cache()
            return

        # offload unnecessary models to cpu
        for model_name in self.model_names:
            if model_name not in load_model_names:
                model = getattr(self, model_name)
                if model is not None and (p := next(model.parameters(), None)) is not None and p.device.type != "cpu":
                    restore_model_from_dict(model, self._offload_param_dict[model_name])
        # load the needed models to device
        for model_name in load_model_names:
            model = getattr(self, model_name)
            if model is None:
                raise ValueError(
                    f"model {model_name} is not loaded, maybe this model has been destroyed by model_lifecycle_finish function with offload_to_disk=True"
                )
            if model is not None and (p := next(model.parameters(), None)) is not None and p.device.type != self.device:
                model.to(self.device)
        # fresh the cuda cache
        empty_cache()

    def model_lifecycle_finish(self, model_names: List[str] | None = None):
        if not self.offload_to_disk or self.offload_mode is None:
            return
        for model_name in model_names:
            model = getattr(self, model_name)
            del model
            if model_name in self._offload_param_dict:
                del self._offload_param_dict[model_name]
            setattr(self, model_name, None)
            print(f"model {model_name} has been deleted from memory")
            logger.info(f"model {model_name} has been deleted from memory")
            empty_cache()

    def compile(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not support compile")
