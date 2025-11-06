import argparse

from diffsynth_engine import WanPipelineConfig
from diffsynth_engine.pipelines import WanVideoPipeline
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.video import save_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the wan speech-to-video pipeline example to run.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument("--parallelism", type=int, default=1, help="Number of parallel devices to use.")
    parser.add_argument("--lora_dir", type=str, default="", help="Directory for LoRA weights.")
    args = parser.parse_args()
    config = WanPipelineConfig.basic_config(
        model_path=fetch_model(
            "Wan-AI/Wan2.2-T2V-A14B",
            revision="bf16",
            path=[
                "high_noise_model/diffusion_pytorch_model-00001-of-00006-bf16.safetensors",
                "high_noise_model/diffusion_pytorch_model-00002-of-00006-bf16.safetensors",
                "high_noise_model/diffusion_pytorch_model-00003-of-00006-bf16.safetensors",
                "high_noise_model/diffusion_pytorch_model-00004-of-00006-bf16.safetensors",
                "high_noise_model/diffusion_pytorch_model-00005-of-00006-bf16.safetensors",
                "high_noise_model/diffusion_pytorch_model-00006-of-00006-bf16.safetensors",
                "low_noise_model/diffusion_pytorch_model-00001-of-00006-bf16.safetensors",
                "low_noise_model/diffusion_pytorch_model-00002-of-00006-bf16.safetensors",
                "low_noise_model/diffusion_pytorch_model-00003-of-00006-bf16.safetensors",
                "low_noise_model/diffusion_pytorch_model-00004-of-00006-bf16.safetensors",
                "low_noise_model/diffusion_pytorch_model-00005-of-00006-bf16.safetensors",
                "low_noise_model/diffusion_pytorch_model-00006-of-00006-bf16.safetensors",
            ],
        ),
        parallelism=args.parallelism,
        device=args.device,
    )
    pipe = WanVideoPipeline.from_pretrained(config)
    pipe.load_loras_high_noise(
        [(f"{args.lora_dir}/wan22-style1-violetevergarden-16-sel-2-high-000100.safetensors", 1.0)],
        fused=False,
        save_original_weight=False,
    )
    pipe.load_loras_low_noise(
        [(f"{args.lora_dir}/wan22-style1-violetevergarden-16-sel-2-low-4-000060.safetensors", 1.0)],
        fused=False,
        save_original_weight=False,
    )

    video = pipe(
        prompt="白天，晴天光，侧光，硬光，暖色调，中近景，中心构图，一个银色短发少女戴着精致的皇冠，穿着华丽的长裙，站在阳光明媚的花园中。她面向镜头微笑，眼睛闪烁着光芒。阳光从侧面照来，照亮了她的银色短发和华丽的服饰，营造出一种温暖而高贵的氛围。微风轻拂，吹动着她裙摆上的蕾丝花边，增添了几分动感。背景是盛开的花朵和绿意盎然的植物，为画面增色不少。,anime style",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_frames=81,
        width=480,
        height=832,
        seed=42,
    )
    save_video(video, "wan22_t2v_lora.mp4", fps=pipe.get_default_fps())

    del pipe
