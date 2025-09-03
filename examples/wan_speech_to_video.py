from PIL import Image
import torch
from diffsynth_engine import WanSpeech2VideoPipelineConfig
from diffsynth_engine.pipelines import WanSpeech2VideoPipeline
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.video import save_video_with_audio


def wan_rs2v(pipe: WanSpeech2VideoPipeline):
    audio_path = "examples/input_s2v/sing.mp3"
    frames = pipe(
        ref_image=Image.open("examples/input_s2v/woman.png").convert('RGB'),
        audio_path=audio_path,
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


def wan_rsp2v(pipe: WanSpeech2VideoPipeline):
    audio_path = "examples/input_s2v/sing.mp3"
    frames = pipe(
        ref_image=Image.open("examples/input_s2v/pose.png").convert('RGB'),
        audio_path=audio_path,
        pose_video_path="examples/input_s2v/pose.mp4",
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


def wan_rs2v_multi_people(pipe: WanSpeech2VideoPipeline):
    audio_path = "examples/input_s2v/sing2.mp3"
    frames = pipe(
        ref_image=Image.open("examples/input_s2v/2girl.png").convert('RGB'),
        audio_path=audio_path,
        void_audio_path="examples/input_s2v/void_audio.mp3",
        prompt="画面清晰，视频中，两个女生正在唱歌，十分深情投入，她们感受着轻柔舒缓的音乐，慢慢摇晃，享受着音乐，表情投入微笑，其中一个女生唱歌，另一个充满深情地看着对方",
        negative_prompt="画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        cfg_scale=5,
        num_inference_steps=40,
        seed=123,
        num_frames_per_clip=80,
        speaking_duration=[[0,6],[1,14],[0,23],[1,100]],
        num_clips=2,
        ref_as_first_frame=False,
    )
    save_video_with_audio(frames, audio_path=audio_path, target_video_path="wan_rs2v_multi_people.mp4")


if __name__ == "__main__":
    # serialization will refuse to proceed if we don't do such here.
    # there seems to be some tensor requiring grad. I have no idea what
    # optimally may need time to search for such tensor, but if we add this no_grad wrapper, we can at least run it.
    with torch.no_grad():
        config = WanSpeech2VideoPipelineConfig.basic_config(
            model_path=fetch_model(
                "Wan-AI/Wan2.2-S2V-14B",
                path=[
                    "diffusion_pytorch_model-00001-of-00004.safetensors",
                    "diffusion_pytorch_model-00002-of-00004.safetensors",
                    "diffusion_pytorch_model-00003-of-00004.safetensors",
                    "diffusion_pytorch_model-00004-of-00004.safetensors",
                ],
            ), 
            parallelism=8,
        )
        config.audio_encoder_dtype = torch.float32
        pipe = WanSpeech2VideoPipeline.from_pretrained(config)
        wan_rs2v(pipe)

        del pipe
