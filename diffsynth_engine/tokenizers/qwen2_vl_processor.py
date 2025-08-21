import os
import re
import torch
import logging
from PIL import Image
from typing import List, Dict, Optional

from diffsynth_engine.tokenizers.qwen2_vl_image_processor import Qwen2VLImageProcessor
from diffsynth_engine.tokenizers.qwen2 import Qwen2TokenizerFast

logger = logging.getLogger(__name__)


class Qwen2VLProcessor:
    def __init__(
        self,
        tokenizer: Qwen2TokenizerFast,
        image_processor: Qwen2VLImageProcessor,
        image_token: str = "<|image_pad|>",
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token = image_token

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_config_path: str | os.PathLike,
        image_processor_config_path: str | os.PathLike,
        **kwargs,
    ):
        tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_config_path)
        image_processor = Qwen2VLImageProcessor.from_pretrained(image_processor_config_path)
        return cls(tokenizer=tokenizer, image_processor=image_processor, **kwargs)

    def batch_decode(
        self,
        ids: List[List[int]] | List[torch.Tensor],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
    ):
        if isinstance(ids[0], torch.Tensor):
            ids = [id_.tolist() for id_ in ids]
        decoded = self.tokenizer.batch_decode(ids, skip_special_tokens, clean_up_tokenization_spaces)
        pattern = r"<\|vision_start\|>.*?<\|vision_end\|>"
        decoded_with_image_tag = [re.sub(pattern, "<image>", d, flags=re.DOTALL) for d in decoded]
        decoded_with_image_tag = [re.sub(r"<\|im_end\|>", "", d) for d in decoded_with_image_tag]
        return decoded_with_image_tag

    def __call__(
        self,
        text: str | List[str],
        images: Optional[List[Image.Image]] = None,
        videos: Optional[List[List[Image.Image]]] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            text (`List[str]`):
                The sequence or batch of sequences to be encoded.
            images (`List[PIL.Image.Image]`):
                The batch of images to be prepared.
            videos (`List[List[PIL.Image.Image]]`):
                The batch of videos to be prepared.
        """
        images_pixel_values, images_grid_thws, video_pixels_values, video_grid_thws = self.image_processor(
            images, videos
        )

        if not isinstance(text, list):
            text = [text]
        if images_grid_thws is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token, "<|placeholder|>" * (images_grid_thws[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)
        text_inputs = self.tokenizer(text, max_length=max_length)

        processed_inputs = text_inputs
        if images_pixel_values is not None:
            processed_inputs["pixel_values"] = torch.from_numpy(images_pixel_values)
        if images_grid_thws is not None:
            processed_inputs["image_grid_thw"] = torch.from_numpy(images_grid_thws)
        if video_pixels_values is not None:
            processed_inputs["pixel_values_videos"] = video_pixels_values
        if video_grid_thws is not None:
            processed_inputs["video_grid_thw"] = video_grid_thws

        return processed_inputs
