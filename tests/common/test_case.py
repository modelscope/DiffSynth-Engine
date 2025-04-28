import unittest
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List
from diffsynth_engine.utils.loader import load_file
from diffsynth_engine.utils.video import save_video, load_video, VideoReader

from tests.common.utils import make_deterministic, compute_normalized_ssim

TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# test flags
RUN_EXTRA_TEST = os.environ.get("RUN_EXTRA_TEST", "")


class TestCase(unittest.TestCase):
    testdata_dir = Path(TEST_ROOT) / "data"

    def setUp(self):
        self.seed = 42
        make_deterministic(self.seed)

    @staticmethod
    def get_device_name() -> str:
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_index)
            return device_name
        else:
            return "cpu"

    @staticmethod
    def get_expect_tensor(name) -> Dict[str, torch.Tensor]:
        return load_file(ImageTestCase.testdata_dir / "expect" / f"{name}")

    @staticmethod
    def get_input_tensor(name) -> Dict[str, torch.Tensor]:
        return load_file(ImageTestCase.testdata_dir / "input" / f"{name}")

    def assertTensorEqual(self, input_tensor: torch.Tensor, expect_tensor: torch.Tensor, atol=1e-3, rtol=1e-3):
        # 计算绝对误差和相对误差
        abs_diff = torch.abs(input_tensor - expect_tensor)
        rel_diff = abs_diff / (torch.abs(expect_tensor))
        # 计算平均绝对误差和相对误差
        mean_abs_diff = torch.mean(abs_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()
        # 计算最大绝对误差和相对误差
        max_abs_diff = torch.max(abs_diff).item()
        max_rel_diff = torch.max(rel_diff).item()

        if not torch.allclose(input_tensor, expect_tensor, atol=atol, rtol=rtol):
            print(f"atol: {atol}\trtol: {rtol}")
            print(f"mean_abs_diff: {mean_abs_diff}\tmean_rel_diff: {mean_rel_diff}")
            print(f"max_abs_diff: {max_abs_diff}\tmax_rel_diff: {max_rel_diff}")

        self.assertTrue(torch.allclose(input_tensor, expect_tensor, atol=atol, rtol=rtol))


class ImageTestCase(TestCase):
    @staticmethod
    def get_expect_image(name) -> Image.Image:
        return Image.open(ImageTestCase.testdata_dir / f"expect/{name}")

    @staticmethod
    def get_input_image(name) -> Image.Image:
        return Image.open(ImageTestCase.testdata_dir / f"input/{name}")

    def assertImageEqual(self, input_image: Image.Image, expect_image: Image.Image, threshold=0.965):
        self.assertGreaterEqual(compute_normalized_ssim(input_image, expect_image), threshold)

    def assertImageEqualAndSaveFailed(self, input_image: Image.Image, expect_image_path: str, threshold=0.965):
        """
        比较input_image和testdata/expect/{name}的SSIM相似度，如果失败则保存input_image到当前工作目录
        """
        try:
            expect_image = self.get_expect_image(expect_image_path)
            self.assertImageEqual(input_image, expect_image, threshold=threshold)
        except Exception as e:
            name = expect_image_path.split("/")[-1]
            input_image.save(f"{name}")
            raise e


class VideoTestCase(TestCase):
    @staticmethod
    def get_input_image(name) -> Image.Image:
        return Image.open(ImageTestCase.testdata_dir / f"input/{name}")

    @staticmethod
    def get_expect_video(name) -> VideoReader:
        return load_video(VideoTestCase.testdata_dir / "expect" / f"{name}")

    @staticmethod
    def get_input_video(name) -> VideoReader:
        return load_video(VideoTestCase.testdata_dir / "input" / f"{name}")

    def save_video(self, video: List[Image.Image], name: str, fps: int = 15):
        save_video(video, name, fps=fps)

    def assertVideoEqual(
        self, input_video: List[Image.Image], expect_video: VideoReader, threshold=0.965, fps: int = 15
    ):
        ssim_list = []
        for i in range(len(input_video)):
            ssim_list.append(compute_normalized_ssim(input_video[i], expect_video[i]))
        ssim_mean = np.mean(ssim_list)
        self.assertGreaterEqual(ssim_mean, threshold)

    def assertVideoEqualAndSaveFailed(
        self, input_video: List[Image.Image], expect_video_path: str, threshold=0.965, fps: int = 15
    ):
        """
        比较input_video和testdata/expect/{name}的SSIM相似度，如果失败则保存input_video到当前工作目录
        """
        try:
            expect_video = self.get_expect_video(expect_video_path)
            self.assertVideoEqual(input_video, expect_video, threshold=threshold)
        except Exception as e:
            name = expect_video_path.split("/")[-1]
            self.save_video(input_video, name, fps=fps)
            raise e
