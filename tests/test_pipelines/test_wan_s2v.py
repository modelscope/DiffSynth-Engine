import unittest

from PIL import Image
import librosa
import torch
from diffsynth_engine import WanSpeech2VideoPipelineConfig
from diffsynth_engine.pipelines import WanSpeech2VideoPipeline
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.video import save_video_with_audio, load_video

from tests.common.test_case import VideoTestCase


class TestWanSpeech2Video(VideoTestCase):
    @classmethod
    def setUpClass(cls):
        config = WanSpeech2VideoPipelineConfig(
            model_path=fetch_model(
                "Wan-AI/Wan2.2-S2V-14B",
                path=[
                    "diffusion_pytorch_model-00001-of-00004.safetensors",
                    "diffusion_pytorch_model-00002-of-00004.safetensors",
                    "diffusion_pytorch_model-00003-of-00004.safetensors",
                    "diffusion_pytorch_model-00004-of-00004.safetensors",
                ],
            ),
        )
        cls.pipe = WanSpeech2VideoPipeline.from_pretrained(config)
        cls.input_data_dir = "tests/data/input/wan_s2v"

    def test_ref_speech_to_video(self):
        audio_path = f"{self.input_data_dir}/sing.mp3"
        audio, _ = librosa.load(audio_path, sr=16000)[0]
        audio = torch.from_numpy(audio)[None]  # (1, audio_len)
        frames = self.pipe(
            ref_image=Image.open(f"{self.input_data_dir}/woman.png").convert("RGB"),
            audio=audio,
            prompt="画面清晰，视频中，一个女人正在唱歌，表情动作十分投入",
            negative_prompt="画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            cfg_scale=4.5,
            num_inference_steps=40,
            seed=42,
            num_frames_per_clip=80,
            num_clips=3,
            ref_as_first_frame=True,
        )
        save_video_with_audio(frames, audio_path=audio_path, target_video_path="wan_rs2v.mp4")

    def test_ref_speech_pose_to_video(self):
        audio_path = f"{self.input_data_dir}/sing.mp3"
        audio, _ = librosa.load(audio_path, sr=16000)
        audio = torch.from_numpy(audio)[None]  # (1, audio_len)
        pose_video = load_video(f"{self.input_data_dir}/pose.mp4")
        frames = self.pipe(
            ref_image=Image.open(f"{self.input_data_dir}/pose.png").convert("RGB"),
            audio=audio,
            pose_video=pose_video.frames,
            pose_video_fps=pose_video.reader.get_meta_data()["fps"],
            prompt="画面清晰，视频中，一个女生正准备开始跳舞，她穿着短裤，她慢慢扭动自己的身体，表情自信阳光，她唱着歌，镜头慢慢拉远",
            negative_prompt="画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            cfg_scale=4.5,
            num_inference_steps=40,
            seed=15250,
            num_frames_per_clip=48,
            num_clips=2,
            ref_as_first_frame=False,
        )
        save_video_with_audio(frames, audio_path=audio_path, target_video_path="wan_rsp2v.mp4")

    def test_ref_speech_to_video_multi_speaker(self):
        audio_path = f"{self.input_data_dir}/sing2.mp3"
        audio, _ = librosa.load(audio_path, sr=16000)
        audio = torch.from_numpy(audio)[None]  # (1, audio_len)
        void_audio, _ = librosa.load(f"{self.input_data_dir}/void_audio.mp3", sr=16000)
        void_audio = torch.from_numpy(void_audio)[None]  # (1, void_audio_len)
        frames = self.pipe(
            ref_image=Image.open(f"{self.input_data_dir}/2girl.png").convert("RGB"),
            audio=audio,
            void_audio=void_audio,
            prompt="画面清晰，视频中，两个女生正在唱歌，十分深情投入，她们感受着轻柔舒缓的音乐，慢慢摇晃，享受着音乐，表情投入微笑，其中一个女生唱歌，另一个充满深情地看着对方",
            negative_prompt="画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            cfg_scale=5,
            num_inference_steps=40,
            seed=123,
            num_frames_per_clip=80,
            speaker_end_sec=[[0, 6], [1, 14], [0, 23], [1, 100]],
            speaker_bbox=[[310, 72, 591, 353], [759, 127, 918, 286]],  # speaker_id: (w_min, h_min, w_max, h_max)
            num_clips=2,
            ref_as_first_frame=False,
        )
        save_video_with_audio(frames, audio_path=audio_path, target_video_path="wan_rs2v_multi_speaker.mp4")


if __name__ == "__main__":
    unittest.main()
