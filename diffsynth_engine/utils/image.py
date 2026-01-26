import torch
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import math
from PIL import Image
from enum import Enum
from typing import List, Tuple, Optional
from torch import Tensor
from torch.nn import functional as F

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


def tensor_to_image(t: torch.Tensor, denormalize: bool = True) -> Image.Image:
    """
    Convert a tensor to an image.
    """
    # b c h w
    if t.dim() == 4:
        t = t[0]
    t = t.permute(1, 2, 0).float().cpu().numpy()
    if denormalize:
        t = (t + 1) / 2
    t = (t.clip(0, 1) * 255).astype(np.uint8)

    if t.shape[2] == 1:
        mode = "L"
        t = t[..., 0]
    elif t.shape[2] == 4:
        mode = "RGBA"
    else:
        mode = "RGB"
    return Image.fromarray(t, mode=mode)


def resize_and_center_crop(image, height: int, width: int):
    resize_operation = transforms.Resize(min(height, width))
    crop_operation = transforms.CenterCrop((height, width))
    return transforms.Compose([resize_operation, crop_operation])(image)


class ChannelDimension(Enum):
    FIRST = "channels_first"
    LAST = "channels_last"


def convert_to_rgb(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, but got {type(image)}")
    if image.mode == "RGB":
        return image
    image = image.convert(mode="RGB")
    return image


def infer_channel_dimension_format(image: np.ndarray) -> ChannelDimension:
    num_channels = (1, 3)
    if image.ndim == 3:
        first_dim, last_dim = 0, 2
    elif image.ndim == 4:
        first_dim, last_dim = 1, 3
    else:
        raise ValueError(f"Unsupported number of image dimensions: {image.ndim}")

    if image.shape[first_dim] in num_channels and image.shape[last_dim] in num_channels:
        logger.warning("Image has both first and last dimensions as channels. This may lead to unexpected behavior.")
        return ChannelDimension.FIRST
    elif image.shape[first_dim] in num_channels:
        return ChannelDimension.FIRST
    elif image.shape[last_dim] in num_channels:
        return ChannelDimension.LAST
    raise ValueError("Unable to infer channel dimension format")


def get_image_size(image: np.ndarray, channel_dim: Optional[ChannelDimension] = None) -> Tuple[int, int]:
    """
    Returns the (height, width) dimensions of the image.
    """
    if channel_dim is None:
        channel_dim = infer_channel_dimension_format(image)
    if channel_dim == ChannelDimension.FIRST:
        return image.shape[-2], image.shape[-1]
    elif channel_dim == ChannelDimension.LAST:
        return image.shape[-3], image.shape[-2]
    else:
        raise ValueError(f"Unsupported channel dimension format: {channel_dim}")


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
) -> Tuple[int, int]:
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    abs_aspect_ratio = max(height, width) / min(height, width)
    if height < factor or width < factor:
        raise ValueError(f"Image height: {height} and width: {width} must be greater than or equal to factor: {factor}")
    elif abs_aspect_ratio > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {abs_aspect_ratio}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def to_channel_dimension_format(
    image: np.ndarray, channel_dim: ChannelDimension, input_channel_dim: Optional[ChannelDimension] = None
) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input image must be of type np.ndarray, got {type(image)}")
    if input_channel_dim is None:
        input_channel_dim = infer_channel_dimension_format(image)
    if input_channel_dim == channel_dim:
        return image
    if channel_dim == ChannelDimension.FIRST:
        image = image.transpose((2, 0, 1))
    elif channel_dim == ChannelDimension.LAST:
        image = image.transpose((1, 2, 0))
    else:
        raise ValueError(f"Unsupported channel dimension format: {channel_dim}")
    return image


def get_channel_dimension_axis(image: np.ndarray, input_data_format: Optional[ChannelDimension] = None) -> int:
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    if input_data_format == ChannelDimension.FIRST:
        return image.ndim - 3
    elif input_data_format == ChannelDimension.LAST:
        return image.ndim - 1
    raise ValueError(f"Unsupported channel dimension format: {input_data_format}")


def rescale_image(
    image: np.ndarray,
    rescale_factor: float,
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[ChannelDimension] = None,
) -> np.ndarray:
    rescaled_image = image.astype(np.float64) * rescale_factor
    if data_format is not None:
        rescaled_image = to_channel_dimension_format(rescaled_image, data_format, input_data_format)
    rescaled_image = rescaled_image.astype(np.float32)
    return rescaled_image


def normalize_image(
    image: np.ndarray,
    mean: List[float],
    std: List[float],
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[ChannelDimension] = None,
) -> np.ndarray:
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    channel_axis = get_channel_dimension_axis(image, input_data_format)
    num_channels = image.shape[channel_axis]
    if len(mean) != num_channels:
        raise ValueError(f"mean must have {num_channels} elements, but got {len(mean)}")
    if len(std) != num_channels:
        raise ValueError(f"std must have {num_channels} elements, but got {len(std)}")
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    if input_data_format == ChannelDimension.LAST:
        image = (image - mean) / std
    else:
        image = ((image.T - mean) / std).T
    if data_format is not None:
        image = to_channel_dimension_format(image, data_format, input_data_format)
    return image


def to_pil_image(
    image: np.ndarray,
    do_rescale: Optional[bool] = None,
    input_data_format: Optional[ChannelDimension] = None,
    image_mode: Optional[str] = None,
) -> Image.Image:
    image = to_channel_dimension_format(image, ChannelDimension.LAST, input_data_format)
    image = np.squeeze(image, axis=-1) if image.shape[-1] == 1 else image
    do_rescale = do_rescale if do_rescale is not None else _need_rescale_pil_conversion(image)
    if do_rescale:
        image = rescale_image(image, 255)
    image = image.astype(np.uint8)
    return Image.fromarray(image, mode=image_mode)


def resize_image(
    image: np.ndarray,
    height: int,
    width: int,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
    reducing_gap: Optional[int] = None,
    input_data_format: Optional[ChannelDimension] = None,
    data_format: Optional[ChannelDimension] = None,
) -> np.ndarray:
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    data_format = data_format if data_format is not None else input_data_format
    do_rescale = _need_rescale_pil_conversion(image)
    pil_image = to_pil_image(image, do_rescale, input_data_format)
    resized_image = pil_image.resize((width, height), resample=resample, reducing_gap=reducing_gap)
    resized_image = np.array(resized_image)
    resized_image = np.expand_dims(resized_image, axis=-1) if resized_image.ndim == 2 else resized_image
    resized_image = to_channel_dimension_format(resized_image, data_format, ChannelDimension.LAST)
    resized_image = rescale_image(resized_image, 1 / 255) if do_rescale else resized_image
    return resized_image


def _need_rescale_pil_conversion(image: np.ndarray) -> bool:
    """
    Detects whether or not the image needs to be rescaled before being converted to a PIL image.
    The assumption is that if the image is of type `np.float` and all values are between 0 and 1, it needs to be
    rescaled.
    """
    if image.dtype == np.uint8:
        do_rescale = False
    elif np.allclose(image, image.astype(int)):
        if np.all(0 <= image) and np.all(image <= 255):
            do_rescale = False
        else:
            raise ValueError(
                "The image to be converted to a PIL image contains value outside the range [0, 255], "
                f"got [{image.min()}, {image.max()}] which cannot be converted to uint8."
            )
    elif np.all(0 <= image) and np.all(image <= 1):
        do_rescale = True
    else:
        raise ValueError(
            "The image to be converted to PIL image contains values outside the range [0, 1]"
            f"got [{image.min()}, {image.max()}] which cannot be converted to uint8."
        )
    return do_rescale


# --------------------------------------------------------------------------------
# Color Alignment Functions
# Based on Li Yi's implementation: https://github.com/pkuliyi2015/sd-webui-stablesr
# --------------------------------------------------------------------------------
def calc_mean_std(feat: Tensor, eps=1e-5):
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def wavelet_blur(image: Tensor, radius: int):
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None]
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image: Tensor, levels=5):
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq


def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor):
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    return content_high_freq + style_low_freq


def adain_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image


def wavelet_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)

    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image
