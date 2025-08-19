# modified from transformers.models.qwen2_vl.image_processing_qwen2_vl
import os
import json
import logging
import numpy as np
from typing import List, Optional
from PIL import Image

from diffsynth_engine.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from diffsynth_engine.utils.image import (
    ChannelDimension,
    convert_to_rgb,
    get_image_size,
    infer_channel_dimension_format,
    rescale_image,
    resize_image,
    smart_resize,
    normalize_image,
    to_channel_dimension_format,
)

logger = logging.getLogger(__name__)


class Qwen2VLImageProcessor:
    def __init__(
        self,
        do_resize: bool = True,
        resample: Image.Resampling = Image.Resampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1.0 / 255,
        do_normalize: bool = True,
        image_mean: List[float] = OPENAI_CLIP_MEAN,
        image_std: List[float] = OPENAI_CLIP_STD,
        do_convert_rgb: bool = True,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ):
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb
        self.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        self.image_mean = image_mean
        self.image_std = image_std
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.temporal_patch_size = temporal_patch_size

    @classmethod
    def from_pretrained(cls, config_file_path: str | os.PathLike, **kwargs):
        init_kwargs = {}
        if not os.path.exists(config_file_path):
            logger.warning(f"Cannot find {config_file_path}, init processor with default parameters")
        else:
            with open(config_file_path, "r", encoding="utf-8") as kwargs_handler:
                init_kwargs = json.load(kwargs_handler)

        init_kwargs.update(**kwargs)
        return cls(**init_kwargs)

    def __call__(
        self,
        images: Image.Image | List[Image.Image],
        videos: Optional[List[List[Image.Image]]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
    ):
        pixel_values, image_grid_thws = None, None
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            pixel_values, image_grid_thws = [], []
            for image in images:
                flatten_patches, image_grid_thw = self._preprocess([image], data_format)
                pixel_values.extend(flatten_patches)
                image_grid_thws.append(image_grid_thw)
            pixel_values = np.array(pixel_values)
            image_grid_thws = np.array(image_grid_thws)

        vision_pixel_values, vision_grid_thws = None, None
        if videos is not None:
            vision_pixel_values, vision_grid_thws = [], []
            for images in videos:
                flatten_patches, video_grid_thw = self._preprocess(images, data_format)
                vision_pixel_values.append(flatten_patches)
                vision_grid_thws.append(video_grid_thw)
            vision_pixel_values = np.array(vision_pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)

        return pixel_values, image_grid_thws, vision_pixel_values, vision_grid_thws

    def _preprocess(self, images: List[Image.Image], data_format: Optional[ChannelDimension] = ChannelDimension.FIRST):
        images = [convert_to_rgb(image) for image in images]
        image_nps = [np.array(image) for image in images]
        input_data_format = infer_channel_dimension_format(image_nps[0])
        height, width = get_image_size(image_nps[0], input_data_format)
        resized_height, resized_width = height, width

        processed_image_nps = []
        for image_np in image_nps:
            if self.do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                image_np = resize_image(
                    image_np, resized_height, resized_width, self.resample, input_data_format=input_data_format
                )

            if self.do_rescale:
                image_np = rescale_image(image_np, self.rescale_factor)

            if self.do_normalize:
                image_np = normalize_image(
                    image_np, self.image_mean, self.image_std, input_data_format=input_data_format
                )
            image_np = to_channel_dimension_format(image_np, data_format, input_data_format)
            processed_image_nps.append(image_np)

        patches = np.array(processed_image_nps)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats = np.repeat(patches[-1][np.newaxis], self.temporal_patch_size - 1, axis=0)
            patches = np.concatenate([patches, repeats], axis=0)
        num_channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            num_channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, num_channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )

        return flatten_patches, (grid_t, grid_h, grid_w)
