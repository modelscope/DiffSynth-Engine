import torch.multiprocessing as mp

from diffsynth_engine.pipelines import WanVideoPipeline, WanModelConfig
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.video import save_video


if __name__ == "__main__":
    mp.set_start_method("spawn")
    config = WanModelConfig(
        model_path=fetch_model("MusePublic/wan2.1-1.3b", path="dit.safetensors"),
        t5_path=fetch_model("muse/wan2.1-umt5", path="umt5.safetensors"),
        vae_path=fetch_model("muse/wan2.1-vae", path="vae.safetensors"),
        dit_fsdp=True,
    )
    pipe = WanVideoPipeline.from_pretrained(config, parallelism=4, use_cfg_parallel=True, offload_mode="cpu_offload")
    pipe.load_lora(
        path=fetch_model("VoidOc/wan_silver", revision="ckpt-15", path="15.safetensors"),
        scale=1.0,
        fused=False,
    )

    video = pipe(
        prompt="一张亚洲女性在水中被樱花环绕的照片。她拥有白皙的肌肤和精致的面容，樱花散落在她的脸颊，她轻盈地漂浮在水中。阳光透过水面洒下，形成斑驳的光影，营造出一种宁静而超凡脱俗的氛围。她的长发在水中轻轻飘动，眼神温柔而宁静，仿佛与周围的自然环境融为一体。背景是淡粉色的樱花花瓣缓缓随着水波上漂浮，增添了一抹梦幻色彩。画面整体呈现出柔和的色调，带有细腻的光影效果。中景脸部人像特写，俯视视角。,wan_silver,wan_silver",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_frames=81,
        width=480,
        height=832,
        seed=42,
    )
    save_video(video, "wan_t2v_lora.mp4", fps=15)

    del pipe
