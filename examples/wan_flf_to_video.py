from PIL import Image

from diffsynth_engine import WanPipelineConfig
from diffsynth_engine.pipelines import WanVideoPipeline
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.video import save_video


if __name__ == "__main__":
    config = WanPipelineConfig.basic_config(
        model_path=fetch_model("MusePublic/wan2.1-flf2v-14b-720p-bf16", path="dit.safetensors"),
        image_encoder_path=fetch_model(
            "muse/open-clip-xlm-roberta-large-vit-huge-14",
            path="open-clip-xlm-roberta-large-vit-huge-14.safetensors",
        ),
        parallelism=4,
        offload_mode="cpu_offload",
    )
    pipe = WanVideoPipeline.from_pretrained(config)

    fisrt_frame = Image.open("input/wan_flf2v_input_first_frame.png").convert("RGB")
    last_frame = Image.open("input/wan_flf2v_input_last_frame.png").convert("RGB")
    # recommmend using resolution 1280x720, 960x960, 720x1280, 832x480, 480x832 for better quality
    video = pipe(
        prompt="CG动画风格，一只蓝色的小鸟从地面起飞，煽动翅膀。小鸟羽毛细腻，胸前有独特的花纹，背景是蓝天白云，阳光明媚。镜跟随小鸟向上移动，展现出小鸟飞翔的姿态和天空的广阔。近景，仰视视角。",
        negative_prompt="镜头切换，镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=[fisrt_frame, last_frame],
        num_frames=81,
        width=960,
        height=960,
        cfg_scale=5.5,
        seed=42,
    )
    save_video(video, "wan_flf2v.mp4", fps=15)

    del pipe
