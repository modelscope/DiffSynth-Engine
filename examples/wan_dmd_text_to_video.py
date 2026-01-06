from diffsynth_engine import WanPipelineConfig
from diffsynth_engine.pipelines import WanDMDPipeline
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.video import save_video


if __name__ == "__main__":
    config = WanPipelineConfig.basic_config(
        model_path=fetch_model(
            "Wan-AI/Wan2.2-T2V-A14B-BF16",
            path=[
                "high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
                "high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
                "high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
                "high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
                "high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
                "high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors",
                "low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
                "low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
                "low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
                "low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
                "low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
                "low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors",
            ],
        ),
        parallelism=1,
    )
    pipe = WanDMDPipeline.from_pretrained(config)
    pipe.load_loras_high_noise(
        [(fetch_model("lightx2v/Wan2.2-Lightning", path="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/high_noise_model.safetensors"), 1.0)],
        fused=False,
    )
    pipe.load_loras_low_noise(
        [(fetch_model("lightx2v/Wan2.2-Lightning", path="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/low_noise_model.safetensors"), 1.0)],
        fused=False,
    )

    video = pipe(
        prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
        num_frames=81,
        width=480,
        height=832,
        seed=42,
        denoising_step_list=[1000, 750, 500, 250],
    )
    save_video(video, "wan_dmd_t2v.mp4", fps=pipe.get_default_fps())

    del pipe
